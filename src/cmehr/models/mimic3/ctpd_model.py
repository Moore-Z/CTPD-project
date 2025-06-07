# CTPD model for mimic3 dataset
from typing import Dict
import ipdb
from einops import rearrange
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from cmehr.models.mimic4.UTDE_modules import multiTimeAttention, gateMLP
# from cmehr.models.mimic4.tslanet_model import PatchEmbed, ICB
# from cmehr.backbone.vision.pretrained import get_biovil_t_image_encoder
# from cmehr.models.mimic4.base_model import MIMIC4LightningModule
from cmehr.models.mimic3.base_model import MIMIC3LightningModule
from cmehr.models.mimic4.mtand_model import Attn_Net_Gated
# from cmehr.utils.hard_ts_losses import hier_CL_hard
# from cmehr.utils.soft_ts_losses import hier_CL_soft
from cmehr.utils.lr_scheduler import linear_warmup_decay
from cmehr.models.common.dilated_conv import DilatedConvEncoder, ConvBlock
from cmehr.models.mimic4.position_encode import PositionalEncoding1D
from cmehr.models.mimic4.UTDE_modules import BertForRepresentation
# from cmehr.models.mimic4.CTPD_model import SlotAttention


class SlotAttention(nn.Module):
    """
    SlotAttention模块:
    用迭代注意力和GRUCell来学习一组slot表示,用于在输入序列(时序/文本)中发现"原型模式"。
    参见论文Cross-Modal Temporal Pattern Discovery中的Prototype学习。
    
    输入:
        slots: [B, num_slots, D], 初始的slot向量
        inputs: [B, T, D], 具体时序/文本特征
    输出:
        slots: [B, num_slots, D], 迭代更新后的slot向量
        attn: [B, num_slots, T], slot对输入序列的注意力分配
    """
    def __init__(self, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        """
        初始化SlotAttention模块
        
        参数:
            dim: slot和输入特征的维度
            iters: 迭代更新slot的次数
            eps: 数值稳定性小值
            hidden_dim: MLP中间层维度
        """
        super().__init__()
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5   # 点乘注意力的缩放系数

        # 将 inputs 投影到 Q, K, V
        self.to_q = nn.Linear(dim, dim)  # 将slots投影为查询Q
        self.to_k = nn.Linear(dim, dim)  # 将inputs投影为键K
        self.to_v = nn.Linear(dim, dim)  # 将inputs投影为值V
        
        # 迭代更新slots时用的 GRUCell
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        # 用于更新slots的MLP    
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        # 若干 LayerNorm 帮助训练稳定
        self.norm_input  = nn.LayerNorm(dim)  # 标准化输入slots
        self.norm_slots  = nn.LayerNorm(dim)  # 迭代中标准化slots
        self.norm_pre_ff = nn.LayerNorm(dim)  # MLP前标准化

    def forward(self, slots, inputs):
        """
        前向传播函数, 迭代更新slots以捕获输入序列中的模式
        
        参数:
            slots: [B, num_slots, D] - 初始slot向量
            inputs: [B, T, D] - 输入特征序列(时序或文本)
            
        返回:
            slots: [B, num_slots, D] - 更新后的slot向量，表示提取的模式
            attn: [B, num_slots, T] - slot对输入序列的注意力权重
        """

        b, n, d = slots.shape  # b: batch_size, n: num_slots, d: 特征维度

        # 对slots做标准化
        slots = self.norm_input(slots)     
        # 将输入投影成 K, V
        k, v = self.to_k(inputs), self.to_v(inputs)  # k,v: [B, T, D]

        for _ in range(self.iters):
            slots_prev = slots  # 保存上一次迭代的slots

            # 对slots做标准化
            slots = self.norm_slots(slots)  # [B, num_slots, D]
            # 将slots投影成Q
            q = self.to_q(slots)  # [B, num_slots, D]

            # 注意力分数 = Q*K^T * scale，计算slots和inputs之间的相似度
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # [B, num_slots, T]
            # softmax并加上 eps，计算注意力权重
            attn = dots.softmax(dim=1) + self.eps  # [B, num_slots, T]

            # 在 num_slots 维度进行归一化，确保每个时间点的权重和为1
            attn = attn / attn.sum(dim=-1, keepdim=True)  # [B, num_slots, T]

            # 计算更新量，基于注意力权重聚合输入特征
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [B, num_slots, D]

            """
            GRUCell: 与 torch.nn.GRU 不同，后者处理完整的时间序列，而 GRUCell 只执行单步计算
            标准输入格式是: new_hidden = gru_cell(input, hidden_state)
            input 形状是 [batch_size, input_size]
            hidden_state 形状是 [batch_size, hidden_size]
            此处并行输入B*n个input来同时更新B*n个hidden_state
            使用updates来更新slots
            """
            slots = self.gru(
                updates.reshape(-1, d),  # 把 [B, n, d] 形状拉直成 [B*n, d] 使得可以输入GRUCell
                slots_prev.reshape(-1, d)  # 把 [B, n, d] 形状拉直成 [B*n, d] 使得可以输入GRUCell
            )

            # 把 [B*n, d] 形状拉回 [B, n, d]
            slots = slots.reshape(b, -1, d)
            # 再加一个 MLP 残差，进一步变换slots
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn
    

class CTPDModule(MIMIC3LightningModule):
    '''
    CTPD模型(Cross-modal Temporal Pattern Discovery)，用于多模态医疗数据处理和预测。
    
    该模型处理不规则时间序列和临床文本数据，通过SlotAttention提取跨模态的时间模式(temporal patterns)，
    将这些模式用于下游任务预测(如院内死亡预测、疾病表型分类等)。
    
    核心步骤:
    1. 使用mTAND将不规则时间序列对齐到参考时间线
    2. 使用BERT编码临床文本笔记
    3. 通过SlotAttention提取和对齐跨模态时间模式
    4. 使用对比学习和重构损失优化模式表示
    5. 融合多模态特征进行下游任务预测
    '''
    def __init__(self,
                 task: str = "ihm", # 任务名称 (如 'ihm', 'pheno' 等)，会影响最后的输出层和 loss
                 orig_d_ts: int = 17, # 不规则时间序列的原始特征维度
                 orig_reg_d_ts: int = 34, # 规则时间序列的原始特征维度
                 warmup_epochs: int = 20, # 学习率预热期epochs数
                 max_epochs: int = 100, # 最大训练轮数
                 ts_learning_rate: float = 4e-5, # 模型学习率
                 embed_time: int = 64, # 时间嵌入的维度, 用于mTAND
                 embed_dim: int = 128, # 时序/文本在整个模型中的通用隐藏维度,对应论文中的"D"
                 num_of_notes: int = 4, # 每个样本中文本笔记的数量
                 period_length: float = 48, # 标准参考时间线长度(小时) --> T_ref
                 num_slots: int = 20, # 原型(prototype)数量，即SlotAttention中的slot数
                 lamb1: float = 1., # 对比损失权重
                 lamb2: float = 1., # 时序重构损失权重
                 lamb3: float = 1., # 文本重构损失权重
                 use_prototype: bool = True, # 是否启用SlotAttention提取原型
                 use_multiscale: bool = True, # 是否提取多尺度时序特征
                 bert_type: str = "prajjwal1/bert-tiny", # 文本编码器类型
                 TS_mixup: bool = True, # 是否启用混合规则/不规则时序特征
                 mixup_level: str = "batch", # 混合级别:'batch'/'batch_seq'/'batch_seq_feature'
                 dropout: float = 0.1, # Dropout率 
                 pooling_type: str = "attention", # 特征池化方式
                 *args,
                 **kwargs
                 ):
        # 初始化父类
        super().__init__(task=task, max_epochs=max_epochs,
                         ts_learning_rate=ts_learning_rate,
                         period_length=period_length)
        self.save_hyperparameters()

        # ============ 超参数赋值 =============
        self.orig_d_ts = orig_d_ts      
        self.orig_reg_d_ts = orig_reg_d_ts
        self.max_epochs = max_epochs
        self.ts_learning_rate = ts_learning_rate
        self.embed_dim = embed_dim
        self.num_of_notes = num_of_notes
        self.tt_max = period_length
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.warmup_epochs = warmup_epochs
        self.use_prototype = use_prototype
        self.use_multiscale = use_multiscale
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level

        # ================= 初始化各类模块 =================

        # ============ 多层时序卷积模块 =============
        # 用于提取不同时间尺度的特征，三层不同膨胀率的卷积
        # [B,D,T_ref]
        self.ts_conv_1 = ConvBlock(
                self.embed_dim,     # 输入维度
                self.embed_dim,     # 输出维度
                kernel_size=3,      # 卷积核大小
                dilation=1,         # 膨胀率=1, 标准卷积
                final=False,        # 非最终层
            )
        self.ts_conv_2 = ConvBlock(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                dilation=2,         # 膨胀率=2，感受野更大
                final=False,
            )
        self.ts_conv_3 = ConvBlock(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                dilation=4,         # 膨胀率=4，感受野最大
                final=True,         # 最终层
            )

        # ============ 文本卷积模块 =============
        # 用于提取文本序列的局部特征
        self.text_conv_1 = ConvBlock(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            dilation=1,
            final=False,
        )

        # ============ 时间嵌入相关层 =============
        # 用于将时间戳转换为连续嵌入向量
        self.periodic = nn.Linear(1, embed_time-1)  # 周期性时间编码(正弦变换)
        self.linear = nn.Linear(1, 1)               # 线性时间编码(保留原始顺序信息)

        # 参考时间线：0到1之间均匀分布的period_length个点
        # 用于将不规则采样对齐到统一参考时间线
        self.time_query = torch.linspace(0, 1., self.tt_max)  # [tt_max]

        # ============ 多时间注意力 mTAND 模块 =============
        # https://mp.weixin.qq.com/s/BWffVjnapdBgtxbXNwITsA
        # 用于不规则时间序列的时间对齐
        self.time_attn_ts = multiTimeAttention(
            self.orig_d_ts*2,  # 输入维度(包含值和掩码)
            self.embed_dim,    # 输出嵌入维度
            embed_time,        # 时间编码维度
            8)                 # 注意力头数
        
        # 用于文本序列的时间对齐
        if bert_type == "prajjwal1/bert-tiny":
            self.time_attn_text = multiTimeAttention(
                128,           # BERT-tiny输出维度
                self.embed_dim, 
                embed_time, 
                8)
        elif bert_type == "yikuan8/Clinical-Longformer":
            self.time_attn_text = multiTimeAttention(
                512,           # Clinical-Longformer输出维度
                self.embed_dim, 
                embed_time, 
                8)

        # ============ 文本编码器 (预训练BERT) ============
        # 加载预训练BERT模型编码文本
        Biobert = AutoModel.from_pretrained(bert_type)
        self.bertrep = BertForRepresentation(bert_type, Biobert)

        # ============ 门控融合 (mixup) ============
        # 用于融合规则和不规则时间序列
        if self.TS_mixup:
            if self.mixup_level == 'batch':
                # 基于batch级别的门控融合
                self.moe_ts = gateMLP(
                    input_dim=self.embed_dim*2,  # 连接两种特征，维度翻倍
                    hidden_size=embed_dim,      
                    output_dim=1,                # 输出一个门控权重
                    dropout=dropout)
            else:
                raise ValueError("Unknown mixedup type")

        # ============ 将规则时序投射到embed_dim维度的1D卷积 ============
        # 将原始规则时序特征转换为统一维度
        # 对特征维度进行卷积，输入shape:[B,D,T]
        self.proj_reg_ts = nn.Conv1d(
            orig_reg_d_ts,     # 输入通道：原始特征维度
            self.embed_dim,    # 输出通道：模型隐藏维度
            kernel_size=1,     # 1x1卷积
            padding=0, 
            bias=False)

        # ============ Prototype相关(SlotAttention) ============
        if self.use_prototype:
            # 位置编码，用于增强序列位置信息
            self.pe = PositionalEncoding1D(embed_dim)
        
            # 定义共享slots(原型)
            self.num_slots = num_slots
            # 原型均值参数，初始化为1x1xD的张量，会扩展为BxKxD
            self.shared_slots_mu = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            # 原型方差参数(对数形式)，用于随机初始化
            self.shared_slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.xavier_uniform_(self.shared_slots_logsigma)

            # 用于时序和文本的SlotAttention模块
            self.ts_grouping = SlotAttention(dim=embed_dim)
            self.text_grouping = SlotAttention(dim=embed_dim)

            # 对比损失权重生成网络
            self.weight_proj = nn.Sequential(
                nn.Linear(int(self.embed_dim * 2), self.num_slots),  # 输入：连接的全局特征
                nn.ReLU(inplace=True),
                nn.Linear(self.num_slots, self.num_slots)            # 输出：每个slot的权重
            )
        
        # ============ 模态融合Transformer ============
        # 用于融合不同模态的特征
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ============ 特征池化层 ============
        # 用于从序列特征中提取全局表示
        self.ts_atten_pooling = Attn_Net_Gated(
            L=embed_dim,    # 输入特征维度
            D=64,           # 中间层维度
            dropout=True,   # 使用dropout
            n_classes=1)    # 输出注意力权重
        self.text_atten_pooling = Attn_Net_Gated(
            L=embed_dim,
            D=64,
            dropout=True,
            n_classes=1)

        # ============ 重构解码器 ============
        # 用于从原型特征重构原始序列，优化原型表示
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.ts_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.ts_proj = nn.Linear(self.embed_dim, self.orig_reg_d_ts)  # 投影回原始特征维度

        # ============ 最终分类/回归输出层 ============
        # 将融合的多模态特征映射到任务输出空间
        self.out_layer = nn.Linear(2 * self.embed_dim, self.num_labels)

        self.train_iters_per_epoch = -1  # 每个epoch的迭代次数，会在训练时设置

    def learn_time_embedding(self, tt):
        """
        将时间戳转换为混合的时间嵌入表示(线性+周期性)
        
        参数:
            tt: [B, T] 或 [T] - 时间戳序列，表示每个数据点的相对时间位置
            
        返回:
            time_embedding: [B, T, embed_time] - 混合的时间嵌入向量
        """
        tt = tt.unsqueeze(-1)  # 扩展维度 [B,T] -> [B,T,1]
        out2 = torch.sin(self.periodic(tt))  # 周期性编码 [B,T,embed_time-1]
        out1 = self.linear(tt)               # 线性编码   [B,T,1]
        return torch.cat([out1, out2], -1)   # 拼接返回   [B,T,embed_time]

    def forward_ts_mtand(self,
                         x_ts: torch.Tensor,  # [B,T,D_irr] 不规则时间序列
                         x_ts_mask: torch.Tensor,  # [B,T,D_irr] 不规则时间序列的掩码
                         ts_tt_list: torch.Tensor):  # [B,T] 不规则时间序列的时间戳
        '''
        使用多时间注意力(mTAND)处理不规则时序，对齐到参考时间线
        
        参数:
            x_ts: [B,T_irr,D_irr] - 不规则时间序列值
            x_ts_mask: [B,T_irr,D_irr] - 不规则时间序列的掩码(1表示有值,0表示缺失)
            ts_tt_list: [B,T_irr] - 不规则时间序列的时间戳
            
        返回:
            proj_x_ts_irg: [T_ref, B, D] - 对齐到参考时间线上的时序表示
        '''

        # 拼接值和掩码，便于mTAND处理
        x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)  # [B,T_irr,2D_irr]
        x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)  # [B,T_irr,2D_irr]
        
        # 将原始时间戳转换为时间嵌入
        time_key_ts = self.learn_time_embedding(ts_tt_list)  # [B,T_irr,embed_time]
        
        # 将参考时间线转换为时间嵌入
        time_query = self.learn_time_embedding(
            self.time_query.unsqueeze(0).type_as(x_ts))  # [1,T_ref,embed_time]

        proj_x_ts_irg = self.time_attn_ts(
            time_query, time_key_ts, x_ts_irg, x_ts_mask)
        
        # 转置维度为 [T_ref,B,D]，以便与其他模块兼容
        proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

        return proj_x_ts_irg
    
    def forward_ts_reg(self, reg_ts: torch.Tensor):
        '''
        处理规则时间序列,投影到嵌入空间对应论文中的applying a 1D convolution to the imputed time series)
        
        参数:
            reg_ts: [B,T_reg,D_reg] - 规则采样的时间序列 (D_reg=34)
            
        返回:
            proj_x_ts_reg: [T_ref,B,D] - 投影后的规则时序特征 (T_ref=T_reg)
        '''
        # 变换维度用于特征卷积 [B,T_reg,D_reg] -> [B,D_reg,T_reg]
        x_ts_reg = reg_ts.transpose(1, 2)
        
        # 使用1x1卷积投影到embed_dim维度D [B,D_reg,T_reg] -> [B,D,T_reg]
        proj_x_ts_reg = self.proj_reg_ts(x_ts_reg)
        
        # 变换维度为 [T_reg,B,D] 以便与其他模块兼容
        proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

        return proj_x_ts_reg

    def gate_ts(self,
                proj_x_ts_irg: torch.Tensor,  # [T_ref,B,D] 不规则时序特征
                proj_x_ts_reg: torch.Tensor):  # [T_ref,B,D] 规则时序特征
        '''
        通过门控机制混合规则和不规则时间序列特征
        
        参数:
            proj_x_ts_irg: [T_ref,B,D] - 不规则时序特征
            proj_x_ts_reg: [T_ref,B,D] - 规则时序特征
            
        返回:
            proj_x_ts: [T_ref,B,D] - 混合后的时序特征
        '''
        assert self.TS_mixup, "TS_mixup is not enabled"
        
        if self.mixup_level == 'batch':
            # 提取全局特征表示(最大池化)
            g_irg = torch.max(proj_x_ts_irg, dim=0).values  # [B,D]
            g_reg = torch.max(proj_x_ts_reg, dim=0).values  # [B,D]
            
            # 拼接两种全局特征
            moe_gate = torch.cat([g_irg, g_reg], dim=-1)  # [B,2*D]
        elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
            # 在特征维度拼接
            moe_gate = torch.cat(
                [proj_x_ts_irg, proj_x_ts_reg], dim=-1)
        else:
            raise ValueError("Unknown mixup type")
            
        # 计算混合权重 (0~1之间)
        mixup_rate = self.moe_ts(moe_gate)  # [B,1] 或根据mixup_level不同形状不同
        
        # 加权混合两种特征
        proj_x_ts = mixup_rate * proj_x_ts_irg + \
            (1 - mixup_rate) * proj_x_ts_reg  # [T_ref,B,D]

        return proj_x_ts

    def forward_text_mtand(self, 
                           input_ids: torch.Tensor,  # [B,N,L] BERT输入ID
                           attention_mask: torch.Tensor,  # [B,N,L] BERT注意力掩码
                           note_time: torch.Tensor,  # [B,N] 文本时间戳
                           note_time_mask: torch.Tensor):  # [B,N] 文本时间掩码
        '''
        处理并对齐文本特征到参考时间线
        
        参数:
            input_ids: [B,N,L] - BERT输入token IDs，N是文本数量，L是序列长度
            attention_mask: [B,N,L] - BERT注意力掩码
            note_time: [B,N] - 每条文本记录的时间戳
            note_time_mask: [B,N] - 文本时间掩码(1表示有效，0表示无效)
            
        返回:
            proj_x_txt: [T_ref,B,D] - 对齐到参考时间线的文本特征
        '''
        
        # 使用BERT编码文本
        x_txt = self.bertrep(input_ids, attention_mask)  # [B,N,D_bert]
        
        # 创建参考时间线的嵌入
        time_query = self.learn_time_embedding(
            self.time_query.unsqueeze(0).type_as(x_txt))  # [1,T_ref,embed_time]
            
        # 将文本时间戳转换为嵌入
        time_key = self.learn_time_embedding(note_time)  # [B,N,embed_time]
        
        # 使用mTAND对齐文本特征到参考时间线
        proj_x_txt = self.time_attn_text(
            time_query, time_key, x_txt, note_time_mask)  # [B,T_ref,D]
            
        # 转置维度为 [T_ref,B,D] 以便与其他模块兼容
        proj_x_txt = proj_x_txt.transpose(0, 1)

        return proj_x_txt

    def infonce_loss(self, sim, temperature=0.07):
        """
        计算InfoNCE对比损失
        
        参数:
            sim: [B,B] - 相似度矩阵，包含batch中每对样本间的相似度
            temperature: float - 温度系数，控制分布的平滑度
            
        返回:
            loss: 标量 - 对比损失值
        """
        sim /= temperature  # 应用温度缩放，较低的温度使分布更加尖锐
        
        # 创建目标标签，对角线元素(正样本对)的索引
        labels = torch.arange(sim.size(0)).type_as(sim).long()

        # 计算交叉熵损失，将相似度矩阵视为每行对应一个样本的logits
        # 正样本对应样本本身(对角线)，其他为负样本
        return F.cross_entropy(sim, labels)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,
                input_ids, attention_mask, note_time, note_time_mask,
                reg_ts, labels=None):
        """
        模型前向传播函数
        
        参数:
            x_ts: [B,T_irr,D_irr] - 不规则时间序列值
            x_ts_mask: [B,T_irr,D_irr] - 不规则时间序列掩码
            ts_tt_list: [B,T_irr] - 不规则时间序列时间戳
            input_ids: [B,N,L] - 文本输入token IDs
            attention_mask: [B,N,L] - 文本注意力掩码
            note_time: [B,N] - 文本时间戳
            note_time_mask: [B,N] - 文本时间掩码
            reg_ts: [B,T_reg,D_reg] - 规则时间序列
            labels: [B,num_labels] 或 [B] - 任务标签(可选)
            
        返回:
            如果有labels:
                loss_dict: 包含各种损失的字典
            如果没有labels:
                输出预测结果，格式取决于任务类型
        """
        
        # STEP 1: 提取并对齐多模态嵌入 
        # ---------------------------------------
        # 1.1 处理不规则时间序列，对齐到参考时间线
        proj_x_ts_irg = self.forward_ts_mtand(
            x_ts, x_ts_mask, ts_tt_list)  # [T_ref,B,D]
            
        # 1.2 处理规则时间序列，投影到嵌入空间
        proj_x_ts_reg = self.forward_ts_reg(reg_ts)  # [T_ref,B,D]
        
        # 1.3 通过门控机制融合规则和不规则时间序列
        proj_x_ts = self.gate_ts(proj_x_ts_irg, proj_x_ts_reg)  # [T_ref,B,D]
        
        # 调整维度从[T,B,D]到[B,D,T]以适应后续卷积操作
        # This code is using the einops library, which provides a very clean and
        # readable way to rearrange tensor dimensions.
        # This line changes the shape of your tensor from [T, B, D] to [B, D, T],
        # which is the format expected by 1D convolution layers (nn.Conv1d) in PyTorch.
        proj_x_ts = rearrange(proj_x_ts, "tt b d -> b d tt")  # [B,D,T_ref]
        
        # 1.4 处理文本序列，对齐到参考时间线
        proj_x_text_irg = self.forward_text_mtand(
            input_ids, attention_mask, note_time, note_time_mask)  # [T_ref,B,D]
            
        # 调整维度以适应后续卷积操作
        proj_x_text = rearrange(proj_x_text_irg, "tt b d -> b d tt")  # [B,D,T_ref]

        # STEP 2: 提取多尺度时序特征
        # ---------------------------------------
        # 通过不同尺度的卷积提取时序特征
        ts_emb_1 = self.ts_conv_1(proj_x_ts)  # [B,D,T_ref] - 第一层卷积特征
        ts_emb_2 = F.avg_pool1d(ts_emb_1, 2)  # [B,D,T_ref/2] - 池化降采样
        ts_emb_2 = self.ts_conv_2(ts_emb_2)   # [B,D,T_ref/2] - 第二层卷积特征(感受野更大)
        ts_emb_3 = F.avg_pool1d(ts_emb_2, 2)  # [B,D,T_ref/4] - 再次池化降采样
        ts_emb_3 = self.ts_conv_3(ts_emb_3)   # [B,D,T_ref/4] - 第三层卷积特征(感受野最大)

        # STEP 3: 收集多尺度特征
        # ---------------------------------------
        ts_feat_list = []
        if not self.use_multiscale:
            # 如果不使用多尺度，只使用第一层特征
            ts_feat = ts_emb_1
            ts_feat = rearrange(ts_feat, "b d tt -> b tt d")  # [B,T_ref,D]
            ts_feat_list.append(ts_feat)
        else:
            # 使用所有三个尺度的特征
            for idx, ts_feat in enumerate([ts_emb_1, ts_emb_2, ts_emb_3]):
                # 调整维度顺序
                ts_feat = rearrange(ts_feat, "b d tt -> b tt d")  # [B,T_i,D] 
                ts_feat_list.append(ts_feat)

        # STEP 4: 提取文本特征
        # ---------------------------------------
        # 对文本进行卷积处理，提取局部特征
        text_emb = self.text_conv_1(proj_x_text)  # [B,D,T_ref]
        text_feat = rearrange(text_emb, "b d tt -> b tt d")  # [B,T_ref,D]
        text_feat_list = []
        text_feat_list.append(text_feat)

        # 拼接所有特征
        ts_feat_concat = torch.cat(ts_feat_list, dim=1)     # [B,sum(T_i),D]
        text_feat_concat = torch.cat(text_feat_list, dim=1) # [B,T_ref,D]
        """
        将这两个模态embedding作为copula输入
        """
        
        # 初始化损失变量
        cont_loss = torch.tensor(0.).type_as(x_ts)
        ts_recon_loss = torch.tensor(0.).type_as(x_ts)
        text_recon_loss = torch.tensor(0.).type_as(x_ts)

        # STEP 5: 使用SlotAttention提取原型特征
        # ---------------------------------------
        if self.use_prototype:
            batch_size = x_ts.size(0)
            
            # 初始化共享原型槽位
            # 使用重参数化技巧(mu + sigma * eps)创建随机初始slots
            shared_mu = self.shared_slots_mu.expand(batch_size, self.num_slots, -1)  # [B,K,D]
            shared_sigma = self.shared_slots_logsigma.exp().expand(batch_size, self.num_slots, -1)  # [B,K,D]
            shared_slots = shared_mu + shared_sigma * torch.randn(shared_mu.shape).type_as(x_ts)  # [B,K,D]
           
            # 为时序特征添加位置编码，增强序列位置信息
            pe = self.pe(ts_feat_concat)  # [B,sum(T_i),D]
            ts_pe = ts_feat_concat + pe  # [B,sum(T_i),D]
            
            # 使用SlotAttention处理时序特征，提取时序原型
            shared_ts_slots, _ = self.ts_grouping(shared_slots, ts_pe)  # [B,K,D]
            shared_ts_slots = F.normalize(shared_ts_slots, dim=-1)  # 规范化
            
            # 为文本特征添加位置编码
            pe = self.pe(text_feat_concat)  # [B,T_ref,D]
            text_pe = text_feat_concat + pe  # [B,T_ref,D]
            
            # 使用SlotAttention处理文本特征，提取文本原型
            shared_text_slots, _ = self.text_grouping(shared_slots, text_pe)  # [B,K,D]
            shared_text_slots = F.normalize(shared_text_slots, dim=-1)  # 规范化

            # STEP 6: 计算对比损失
            # ---------------------------------------
            # 计算全局特征表示(每个模态所有原型的平均)
            global_ts_feat = shared_ts_slots.mean(dim=1)    # [B,D]
            global_text_feat = shared_text_slots.mean(dim=1)  # [B,D]
            
            # 连接全局特征
            global_feat = torch.cat([global_ts_feat, global_text_feat], dim=1)  # [B,2D]
            
            # 计算每个原型对的权重
            slot_weights = F.softmax(self.weight_proj(global_feat), dim=-1)  # [B,K]
            
            # 计算每对样本间的原型相似度矩阵
            pair_similarity = torch.einsum('bnd,cnd->bcn', shared_ts_slots, shared_text_slots)  # [B,B,K]
            
            # 加权聚合得到最终样本间相似度
            similarity = torch.einsum('bcn,bn->bc', pair_similarity, slot_weights)  # [B,B]
            
            # 计算InfoNCE对比损失
            cont_loss = self.infonce_loss(similarity, temperature=0.2)

            # STEP 7: 计算重构损失
            # ---------------------------------------
            # 准备TS的“原型表示”特征
            concat_ts_slot = shared_ts_slots  # [B,K,D]
            # 准备TS的“时间戳级别表示”特征
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)  # [B,sum(T_i),D]
            
            # 准备TEXT的“原型表示”特征
            concat_text_slot = shared_text_slots  # [B,K,D]
            # 准备TEXT的“时间戳级别表示”特征
            concat_text_feat = torch.cat(text_feat_list, dim=1)  # [B,T_ref,D]
            
            # 将所有特征拼接在一起，用于融合和预测
            # 包含原型特征和原始时间步级特征
            concat_feat = torch.cat([concat_ts_slot, concat_ts_feat,
                                     concat_text_slot, concat_text_feat], dim=1)  # [B,K*2+sum(T_i)+T_ref,D]

            ## 下面想通过学习到的“TS原型槽位（slots）”去重构原始规则时间序列，从而让模型学到更高质量的跨模态时间模式
            """
            类比transformer的“Teacher Forcing”训练思路, 也就是在训练自回归序列模型时，每一步解码都能看到前一步的“真值”输入
            训练时并没有让模型用自己预测出来的东西当输入，而是用到了真实序列的片段当输入去预测下一个时间步
            WHY:
            Teacher Forcing 训练收敛更稳定、更快，因为模型随时能得到“正确的历史信息”作为输入，而不会被早期未训练好时的错误输出所干扰。
            """
            # 准备重建的规则时间序列
            ts_tgt_embs = rearrange(proj_x_ts_reg, "tt b d -> b tt d")  # [B,T_ref,D]
            
            # 创建因果掩码，确保只使用过去信息进行预测
            mask = nn.Transformer.generate_square_subsequent_mask(self.tt_max - 1)
            
            # 使用Transformer解码器进行重构
            pred_ts_emb = self.ts_decoder(
                tgt=ts_tgt_embs[:, :-1],   # 将除最后一个时间步外的目标序列作为输入(类比transformer decoder的输入)
                memory=concat_ts_slot,     # 原型记忆(提供上下文) (类比transformer encoder的输出, 用于cross attention)
                tgt_mask=mask,             # 因果掩码(类比transformer decoder的mask)
                tgt_is_causal=True)        # [B,T_ref-1,D]
            
            # 投影回原始特征空间
            pred_ts = self.ts_proj(pred_ts_emb)  # [B,T_ref-1,D_reg]
            
            # 计算MSE重构损失
            ts_recon_loss = F.mse_loss(pred_ts, reg_ts[:, 1:])  # 预测除第一个时间步外的序列

            ## 下面想通过学习到的“TEXT原型槽位（slots）”去重构原始文本序列，从而让模型学到更高质量的跨模态时间模式
            # 准备重建的文本序列
            text_tgt_embs = rearrange(proj_x_text_irg, "tt b d -> b tt d")  # [B,T_ref,D]
            
            # 使用Transformer解码器重构文本
            pred_text_emb = self.text_decoder(
                tgt=text_tgt_embs[:, :-1],  # 将除最后一个时间步外的目标序列作为输入(类比transformer decoder的输入)
                memory=concat_text_slot,    # 文本原型记忆(类比transformer encoder的输出, 用于cross attention)
                tgt_mask=mask,              # 因果掩码(类比transformer decoder的mask)
                tgt_is_causal=True)         # [B,T_ref-1,D]
                
            
            # 计算文本重构损失
            text_recon_loss = F.mse_loss(pred_text_emb, text_tgt_embs[:, 1:])
        else:
            # 如果不使用原型，直接使用时间步级特征
            concat_ts_feat = torch.cat(ts_feat_list, dim=1)  # [B,sum(T_i),D]
            concat_text_feat = torch.cat(text_feat_list, dim=1)  # [B,T_ref,D]
            concat_feat = torch.cat([concat_ts_feat, concat_text_feat], dim=1)  # [B,sum(T_i)+T_ref,D]

        # STEP 8: 多模态特征融合
        # ---------------------------------------
        # 使用Transformer编码器融合所有特征 (sum_tokens = K*2+sum(T_i)+T_ref)
        fusion_feat = self.fusion_layer(concat_feat)  # [B,sum_tokens,D]

        # STEP 9: 特征池化与最终预测
        # ---------------------------------------
        if self.use_prototype:
            # 计算时序特征的token数量
            num_ts_tokens = concat_ts_slot.size(1) + concat_ts_feat.size(1)  # K + sum(T_i)
        else:
            num_ts_tokens = concat_ts_feat.size(1)  # sum(T_i)
        
        # 分别对两种模态的特征进行平均池化 (Attention Pooling)
        last_ts_feat = torch.mean(fusion_feat[:, :num_ts_tokens, :], dim=1)  # [B,D]
        last_text_feat = torch.mean(fusion_feat[:, num_ts_tokens:, :], dim=1)  # [B,D]
        
        # 拼接两种模态的特征 (L_Pred)
        last_hs = torch.cat([last_ts_feat, last_text_feat], dim=1)  # [B,2D]

        # 生成最终预测
        output = self.out_layer(last_hs)  # [B,num_labels]

        # STEP 10: 计算损失并返回结果
        # ---------------------------------------
        # 收集各种损失
        loss_dict = {
            "cont_loss": cont_loss,                  # 对比损失
            "ts_recon_loss": ts_recon_loss,          # 时序重构损失
            "text_recon_loss": text_recon_loss,      # 文本重构损失
        }
        
        # 根据不同任务类型处理输出和计算损失
        if self.task in ['ihm', 'readm']:
            # 在院死亡或再入院预测任务(二分类)
            if labels != None:
                # 训练模式:计算交叉熵损失
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                # 加权组合所有损失
                loss_dict["total_loss"] = ce_loss + cont_loss * self.lamb1 + ts_recon_loss * self.lamb2 + text_recon_loss * self.lamb3
                return loss_dict
            # 推理模式:返回正类概率
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            # 表型分类任务(多标签)
            if labels != None:
                # 训练模式:计算多标签交叉熵损失
                labels = labels[:, 1:]
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                # 加权组合所有损失
                loss_dict["total_loss"] = ce_loss + cont_loss * self.lamb1 + ts_recon_loss * self.lamb2 + text_recon_loss * self.lamb3
                return loss_dict
            # 推理模式:返回sigmoid概率
            return torch.sigmoid(output)
        
        elif self.task == 'los':
            # 住院时长预测任务(回归)
            if labels != None:
                # 训练模式:计算MSE损失
                labels = labels.float()
                # 确保输出和标签形状匹配
                if output.shape != labels.shape:
                    if len(labels.shape) == 1:
                        labels = labels.unsqueeze(1)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(1)
                
                # 计算MSE损失
                mse_loss = self.loss_fct1(output, labels)
                loss_dict["mse_loss"] = mse_loss
                # 加权组合所有损失
                loss_dict["total_loss"] = mse_loss + cont_loss * self.lamb1 + ts_recon_loss * self.lamb2 + text_recon_loss * self.lamb3
                return loss_dict
            # 推理模式:返回回归值
            return output.squeeze(-1)

    def configure_optimizers(self):
        """
        配置模型优化器，使用Adam优化器并对不同参数组设置不同学习率
        
        - 为预训练BERT参数使用较小的学习率
        - 为其他模型参数使用标准学习率
        
        返回:
            optimizer: 配置好的Adam优化器
        """
        optimizer= torch.optim.Adam([
                # 非BERT参数组：使用标准学习率
                {'params': [p for n, p in self.named_parameters() if 'bert' not in n]},
                # BERT参数组：使用较小学习率以微调
                {'params':[p for n, p in self.named_parameters() if 'bert' in n], 'lr': self.ts_learning_rate}
            ], lr=self.ts_learning_rate)
        
        return optimizer
    
        # 下面是使用学习率预热调度器的代码，已被注释
        # assert self.train_iters_per_epoch != -1, "train_iters_per_epoch is not set"
        # warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        # total_steps = self.train_iters_per_epoch * self.max_epochs

        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         linear_warmup_decay(warmup_steps, total_steps, cosine=True),
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        # return [optimizer], [scheduler]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cmehr.paths import *
    # from cmehr.dataset.mimic4_pretraining_datamodule import MIMIC4DataModule
    from cmehr.dataset.mimic3_downstream_datamodule import MIMIC3DataModule

    datamodule = MIMIC3DataModule(
        file_path=str("C:/Users/zhumo/Dataset/MIMIC3/EHR_data/mimiciii_benchmark/output_mimic3/ihm"),
        tt_max=24,
        bert_type="prajjwal1/bert-tiny",
        max_length=512
    )
    batch = dict()
    for batch in datamodule.val_dataloader():
        break
    for k, v in batch.items():
        print(f"{k}: ", v.shape)

    """
    ts: torch.Size([4, 157, 17])
    ts_mask:  torch.Size([4, 157, 17])
    ts_tt:  torch.Size([4, 157])
    reg_ts:  torch.Size([4, 48, 34])
    input_ids:  torch.Size([4, 5, 128])
    attention_mask:  torch.Size([4, 5, 128])
    note_time:  torch.Size([4, 5])
    note_time_mask: torch.Size([4, 5])
    label: torch.Size([4])
    """
    model = CTPDModule(
        period_length=24,
        task="los",
        use_multiscale=True,
        use_prototype=True,
    )
    loss = model(
        # x_ts, x_ts_mask, ts_tt_list

        x_ts=batch["ts"],
        x_ts_mask=batch["ts_mask"],
        ts_tt_list=batch["ts_tt"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        note_time=batch["note_time"],
        note_time_mask=batch["note_time_mask"],
        reg_ts=batch["reg_ts"],
        labels=batch["label"]
    )
    print(loss)

    # lstm_decoder = AttnDecoder(24, 128, 128, 17)
    # input_encoded = torch.randn(4, 34, 128)
    # need a weights: 4, 34, 17
    # y_history = torch.randn(4, 24, 17)
    # out = lstm_decoder(input_encoded, y_history)
    # print(out.shape)

    # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
    # transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
    # memory = torch.rand(32, 10, 512)
    # tgt = torch.rand(32, 20, 512)
    # out = transformer_decoder(tgt, memory)
    # ipdb.set_trace()