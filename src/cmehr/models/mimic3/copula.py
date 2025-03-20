# CTPD model for mimic3 dataset
from typing import Dict
import ipdb
from einops import rearrange
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.distributions import MultivariateNormal

from scipy.stats import beta

from cmehr.models.mimic3.mvnorm import multivariate_normal_cdf as Phi

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


class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv


class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence 
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)

        return loss_rank


class CosineLoss(nn.Module):
    
    def forward(self, cxr, ehr):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        
        return loss


class CopulaLoss(nn.Module):

    def __init__(self, dim=256, K=3, rho_scale=-5, family="Gumbel"):
        super(CopulaLoss, self).__init__()
        """
        Copula损失函数：实现CM2论文中的变分Copula模型
        
        为多模态数据建模联合分布：
        1. 使用高斯混合模型(GMM)建模每个模态的边缘分布
        2. 使用Copula函数建模模态间的依赖关系
        
        参数:
            dim: 特征维度
            K: GMM的组件数量
            rho_scale: 相关系数缩放因子
            family: Copula族的类型，支持Gumbel、Clayton、Gaussian和Frank
        """
        # Copula参数(θ)，控制模态间的依赖强度
        self.theta = nn.Parameter(torch.ones(1) * 1)

        # GMM混合权重，对应论文中的π_m
        self.pi_x = nn.Parameter(torch.ones([K]) / K)
        self.pi_y = nn.Parameter(torch.ones([K]) / K)

        # 选择适当的Copula族
        if family == "Gumbel":
            self.copula_cdf = self.gumbel_cdf
            self.copula_pdf = self.gumbel_cdf # TODO: Use the correct pdf later...
        elif family == "Clayton":
            self.copula_cdf = self.clayton_cdf
            self.copula_pdf = self.clayton_pdf
        elif family == "Gaussian":
            self.copula_cdf = self.gaussian_copula_cdf
            self.copula_pdf = self.gaussian_copula_pdf
        elif family == "Frank":
            self.copula_cdf = self.frank_cdf
            self.copula_pdf = self.frank_pdf

        # GMM的均值参数，对应论文中的μ_m,k
        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        # GMM的协方差参数，对应论文中的Σ_m,k
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * -4)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * -4)
        self.K = K

    def forward(self, x, y):
        """
        计算ELBO损失函数
        
        参数:
            x: 模态1的特征表示
            y: 模态2的特征表示
            
        返回:
            负对数似然(负ELBO)，用于最小化训练目标
        """
        # 获取GMM的对数混合权重
        pi_x = self.pi_x.log_softmax(dim=-1)
        pi_y = self.pi_y.log_softmax(dim=-1)

        # 获取GMM的协方差矩阵（确保正定性）
        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # 计算边缘分布的CDF，对应论文中的F_m(z_m)
        log_u_list = [self.mvn_cdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        log_v_list = [self.mvn_cdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        
        # 计算Copula密度函数，对应论文中的c(F_1(z_1),...,F_M(z_M))
        c_list = [self.copula_pdf(u, v) for u, v in zip(log_u_list, log_v_list)]

        # 计算边缘分布的PDF，对应论文中的f_m(z_m)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        
        # 计算联合分布的对数似然，对应论文中的ELBO计算
        # log p(x,y) = log[c(F_x(x), F_y(y)) * f_x(x) * f_y(y)]
        loss = torch.stack(c_list, dim=1) + torch.stack(u_log_pdf, dim=1) + pi_x + torch.stack(v_log_pdf, dim=1) + pi_y
        loss = torch.logsumexp(loss, -1).mean(0)

        # 检查数值稳定性
        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(c_list, dim=1)).any()

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = 负对数似然，用于最小化训练目标

    def rsample(self, n_samples=[0]):
        """
        从学习的GMM分布中进行采样，用于处理缺失模态
        
        参数:
            n_samples: 采样数量
            
        返回:
            采样的特征表示
        """
        # 获取GMM的混合权重
        pi_x = self.pi_x.softmax(dim=-1).clamp(min=1e-15)
        pi_y = self.pi_y.softmax(dim=-1).clamp(min=1e-15)

        # 获取GMM的协方差
        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # 使用重参数化技巧从GMM中采样
        assert not torch.isnan(self.mu_x).any()
        assert not torch.isnan(cov_x).any()
        assert not torch.isnan(pi_x).any()
        assert not torch.isnan(self.mu_y).any()
        assert not torch.isnan(cov_y).any()
        assert not torch.isnan(pi_y).any()
        
        # 从每个高斯组件采样并加权组合
        x_samples = [MultivariateNormal(self.mu_x[k], scale_tril=torch.diag(cov_x[k])).rsample(sample_shape=n_samples) * pi_x[k] for k in range(self.K)]
        y_samples = [MultivariateNormal(self.mu_y[k], scale_tril=torch.diag(cov_y[k])).rsample(sample_shape=n_samples) * pi_y[k] for k in range(self.K)]

        x_samples = torch.stack(x_samples, dim=0).sum(dim=0)
        y_samples = torch.stack(y_samples, dim=0).sum(dim=0)
        
        # 检查数值稳定性
        assert not torch.isnan(x_samples).any()
        assert not torch.isnan(y_samples).any()

        assert not torch.isinf(x_samples).any()
        assert not torch.isinf(y_samples).any()

        return y_samples

    def gumbel_cdf(self, log_u, log_v):
        """
        Gumbel Copula的累积分布函数
        
        Gumbel Copula适合建模正尾部依赖关系，即极大值的相关性
        
        参数:
            log_u: 第一个模态的CDF
            log_v: 第二个模态的CDF
            
        返回:
            Copula CDF的对数值
        """
        theta = self.theta.clamp(min=1)  # θ ≥ 1确保合法的Copula

        g = (-log_u) ** theta + (-log_v) ** theta
        log_copula_cdf = - g ** (1 / theta)

        return log_copula_cdf

    def gumbel_density(self, u, v):
        """
        Density of Bivariate Gumbel Copula
        """
        g = (- torch.log(u)) ** self.theta + (- torch.log(v)) ** self.theta
        copula_cdf = torch.exp(- g ** (1 / self.theta))
        density = g ** (2 * (1 - self.theta) / self.theta) * ((self.theta - 1) * g ** (- 1 / self.theta) + 1)
        density *= copula_cdf
        density *= (- torch.log(u)) ** (self.theta - 1) * (- torch.log(v)) ** (self.theta - 1)
        density /= u * v

        return density

    def clayton_cdf(self, log_u, log_v):
        alpha = self.theta.clamp(1e-5)

        log_copula_cdf = torch.exp(-alpha * log_u) + torch.exp(-alpha * log_v) - 1
        log_copula_cdf = torch.log(log_copula_cdf.clamp())

        return log_copula_cdf

    def clayton_pdf(self, log_u, log_v):
        pass

    def frank_cdf(self, log_u, log_v):
        theta = self.theta

        u = torch.exp(log_u)
        v = torch.exp(log_v)

        cdf = torch.exp(-theta * u - 1) * torch.exp(-theta * v - 1)
        cdf /= torch.exp(-theta - 1)
        cdf += 1
        cdf = - torch.log(cdf.clamp(1e-9)) / theta
        cdf = torch.log(cdf.clamp(1e-9))

        assert not torch.isnan(cdf).any()

        return cdf


    def frank_pdf(self, log_u, log_v):
        theta = self.theta

        # Performed some standardization
        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2)

        pdf = (torch.exp(-theta) - 1) * -theta * torch.exp(-theta * (u + v))
        pdf = torch.log(pdf.clamp(1e-5))
        pdf -= 2 * torch.logsumexp(torch.stack([
            -theta * torch.ones_like(u),
            -theta * u,
            -theta * v,
            -theta * (u+v)
        ]), dim=0)

        return pdf

    def gaussian_copula_cdf(self, log_u, log_v):
        pass

    def gaussian_copula_pdf(self, log_u, log_v):
        rho = torch.tanh(self.theta)

        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2).clamp(min=1e-6, max=0.99999)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2).clamp(min=1e-6, max=0.99999)

        a = np.sqrt(2) * torch.erfinv(2 * u - 1)
        b = np.sqrt(2) * torch.erfinv(2 * v - 1)

        assert not torch.isinf(a).any()
        assert not torch.isinf(b).any()

        log_pdf = - ((a ** 2 + b ** 2) * rho ** 2 - 2 * a * b * rho) / (2 * (1 - rho ** 2))
        log_pdf -= 0.5 * torch.log((1 - rho ** 2).clamp(min=0.001))
        return log_pdf

    def mvn_cdf(self, x, mu, cov):
        """
        Log CDF of multivariate normal distribution
        """
        m = mu - x  # actually do P(Y-value<0)
        m_shape = m.shape
        d = m_shape[-1]
        z = -m / cov
        q = (torch.erfc(-z*0.70710678118654746171500846685)/2)
        q = q.clamp(min=1e-15)
        phi = torch.log(q).sum(-1)
        phi = phi.clamp(max=phi[phi < 0].max(-1)[0])

        return phi
        # return Phi(x, mu, cov)

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_log_pdf(self, x):
        """
        PDF of multivariate normal distribution
        """

        return (-torch.log(torch.sqrt(2 * torch.pi))
                - torch.log(self.std_dev)
                - ((x - self.mu) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)
    

class CopulaModule(MIMIC3LightningModule):
    '''
    Copula模型，用于多模态医疗数据处理和预测。
    
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
                 use_multiscale: bool = True, # 是否提取多尺度时序特征
                 bert_type: str = "prajjwal1/bert-tiny", # 文本编码器类型
                 TS_mixup: bool = True, # 是否启用混合规则/不规则时序特征
                 mixup_level: str = "batch", # 混合级别:'batch'/'batch_seq'/'batch_seq_feature'
                 dropout: float = 0.1, # Dropout率 
                 pooling_type: str = "attention", # 特征池化方式
                 # ------Copula参数------
                 lamb_copula: float = 0.00001,  # 新增的copula损失权重
                 K: int = 3,              # GMM组件数量
                 rho_scale: float = -5,  # 相关系数缩放因子
                 copula_family: str = "Gumbel",  # Copula族类型
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
        self.lamb_copula = lamb_copula  # 新增的copula损失权重
        self.warmup_epochs = warmup_epochs
        self.use_multiscale = use_multiscale
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.TS_mixup = TS_mixup
        self.mixup_level = mixup_level
        
        # Copula参数
        self.K = K
        self.rho_scale = rho_scale
        self.copula_family = copula_family

        # ================= 初始化各类模块 =================

        # ============ 多层时序卷积模块 =============
        # 用于提取不同时间尺度的特征，三层不同膨胀率的卷积
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

        # ============ 最终分类/回归输出层 ============
        # 将融合的多模态特征映射到任务输出空间
        self.out_layer = nn.Linear(2 * self.embed_dim, self.num_labels)

        self.train_iters_per_epoch = -1  # 每个epoch的迭代次数，会在训练时设置

        # ============ LSTM编码器 ============
        # 用于将时序特征编码为固定维度向量
        self.ts_lstm = nn.LSTM(
            input_size=self.embed_dim,      # 输入维度
            hidden_size=self.embed_dim,     # 隐藏状态维度 D
            num_layers=1,                   # LSTM层数
            batch_first=True,               # 批次在第一维
            bidirectional=False             # 单向LSTM
        )
        
        # 用于将文本特征编码为固定维度向量
        self.text_lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # ============ Copula模块 (替代原来的Prototype) ============
        # 初始化Copula损失函数
        self.copula_loss = CopulaLoss(
            dim=embed_dim,
            K=K, 
            rho_scale=rho_scale, 
            family=copula_family
        )

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
        # 1.1 处理不规则时间序列，对齐到参考时间线
        proj_x_ts_irg = self.forward_ts_mtand(
            x_ts, x_ts_mask, ts_tt_list)  # [T_ref,B,D]
            
        # 1.2 处理规则时间序列，投影到嵌入空间
        proj_x_ts_reg = self.forward_ts_reg(reg_ts)  # [T_ref,B,D]
        
        # 1.3 通过门控机制融合规则和不规则时间序列
        proj_x_ts = self.gate_ts(proj_x_ts_irg, proj_x_ts_reg)  # [T_ref,B,D]
        
        # 调整维度从[T,B,D]到[B,D,T]以适应后续卷积操作
        proj_x_ts = rearrange(proj_x_ts, "tt b d -> b d tt")  # [B,D,T_ref]
        
        # 1.4 处理文本序列，对齐到参考时间线
        proj_x_text_irg = self.forward_text_mtand(
            input_ids, attention_mask, note_time, note_time_mask)  # [T_ref,B,D]
            
        # 调整维度以适应后续卷积操作
        proj_x_text = rearrange(proj_x_text_irg, "tt b d -> b d tt")  # [B,D,T_ref]

        # STEP 2: 提取多尺度时序特征
        # 通过不同尺度的卷积提取时序特征
        ts_emb_1 = self.ts_conv_1(proj_x_ts)  # [B,D,T_ref] - 第一层卷积特征
        ts_emb_2 = F.avg_pool1d(ts_emb_1, 2)  # [B,D,T_ref/2] - 池化降采样
        ts_emb_2 = self.ts_conv_2(ts_emb_2)   # [B,D,T_ref/2] - 第二层卷积特征(感受野更大)
        ts_emb_3 = F.avg_pool1d(ts_emb_2, 2)  # [B,D,T_ref/4] - 再次池化降采样
        ts_emb_3 = self.ts_conv_3(ts_emb_3)   # [B,D,T_ref/4] - 第三层卷积特征(感受野最大)

        # STEP 3: 收集多尺度特征
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
        # 对文本进行卷积处理，提取局部特征
        text_emb = self.text_conv_1(proj_x_text)  # [B,D,T_ref]
        text_feat = rearrange(text_emb, "b d tt -> b tt d")  # [B,T_ref,D]
        text_feat_list = []
        text_feat_list.append(text_feat)

        # 拼接所有特征
        ts_feat_concat = torch.cat(ts_feat_list, dim=1)     # [B,sum(T_i),D]
        text_feat_concat = torch.cat(text_feat_list, dim=1) # [B,T_ref,D]
        
        # 使用LSTM编码时序特征
        _, (ts_hidden, _) = self.ts_lstm(ts_feat_concat)  # ts_hidden: [1, B, D]
        ts_encoded = ts_hidden.squeeze(0)  # [B, D]
        
        # 使用LSTM编码文本特征
        _, (text_hidden, _) = self.text_lstm(text_feat_concat)  # text_hidden: [1, B, D]
        text_encoded = text_hidden.squeeze(0)  # [B, D]

        # STEP 5: 计算copula损失
        copula_loss = self.copula_loss(ts_encoded, text_encoded)

        # STEP 6: 如果不使用原型，直接使用时间步级特征
        concat_ts_feat = torch.cat(ts_feat_list, dim=1)  # [B,sum(T_i),D]
        concat_text_feat = torch.cat(text_feat_list, dim=1)  # [B,T_ref,D]
        concat_feat = torch.cat([concat_ts_feat, concat_text_feat], dim=1)  # [B,sum(T_i)+T_ref,D]

        # STEP 8: 多模态特征融合
        # 使用Transformer编码器融合所有特征 (sum_tokens = K*2+sum(T_i)+T_ref)
        fusion_feat = self.fusion_layer(concat_feat)  # [B,sum_tokens,D]

        # STEP 9: 特征池化与最终预测
        num_ts_tokens = concat_ts_feat.size(1)  # sum(T_i)
        
        # 分别对两种模态的特征进行平均池化
        last_ts_feat = torch.mean(fusion_feat[:, :num_ts_tokens, :], dim=1)  # [B,D]
        last_text_feat = torch.mean(fusion_feat[:, num_ts_tokens:, :], dim=1)  # [B,D]
        
        # 拼接两种模态的特征
        last_hs = torch.cat([last_ts_feat, last_text_feat], dim=1)  # [B,2D]

        # 生成最终预测
        output = self.out_layer(last_hs)  # [B,num_labels]

        # STEP 8: 计算损失并返回结果
        # 收集损失
        loss_dict = {
            "copula_loss": copula_loss,
        }
        
        # 根据不同任务类型处理输出和计算损失
        if self.task in ['ihm', 'readm']:
            # 在院死亡或再入院预测任务(二分类)
            if labels != None:
                # 训练模式:计算交叉熵损失
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                # 加权组合损失
                loss_dict["total_loss"] = ce_loss + copula_loss * self.lamb_copula
                return loss_dict
            # 推理模式:返回正类概率
            return F.softmax(output, dim=-1)[:, 1]

        elif self.task == 'pheno':
            # 表型分类任务(多标签)
            if labels != None:
                # 训练模式:计算多标签交叉熵损失
                labels = labels.float()
                ce_loss = self.loss_fct1(output, labels)
                loss_dict["ce_loss"] = ce_loss
                # 加权组合损失
                loss_dict["total_loss"] = ce_loss + copula_loss * self.lamb_copula
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
                # 加权组合损失
                loss_dict["total_loss"] = mse_loss + copula_loss * self.lamb_copula
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
        file_path=str("/Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3/ihm"),
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
    model = CopulaModule(
        task="ihm",
        use_multiscale=True,
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