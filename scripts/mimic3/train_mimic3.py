'''
这是在 MIMIC-III 数据集上运行时序(TS)实验的脚本。
'''

# 导入必要的库
from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import ipdb
import wandb
import os
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from cmehr.dataset import MIMIC3DataModule
# 导入各种模型模块
# from cmehr.models.mimic4 import (
#     CNNModule, ProtoTSModel, IPNetModule, GRUDModule, SEFTModule,
#     MTANDModule, DGM2OModule, MedFuseModule, UTDEModule, LSTMModule)
from cmehr.models.mimic3.ctpd_model import CTPDModule
from cmehr.models.mimic3.copula import CopulaModule
from cmehr.paths import *

# 设置PyTorch后端配置，提高性能和稳定性
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

# 创建命令行参数解析器
parser = ArgumentParser(description="PyTorch Lightning EHR Model")
# 设置任务类型，可选ihm(院内死亡预测)、decomp(病情恶化预测)、los(住院时长预测)、pheno(表型识别)、readm(再入院预测)
parser.add_argument("--task", type=str, default="pheno",
                    choices=["ihm", "decomp", "los", "pheno", "readm"])
parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
parser.add_argument("--num_workers", type=int, default=4)   # 数据加载的工作进程数
parser.add_argument("--update_counts", type=int, default=3) # 更新次数
parser.add_argument("--max_epochs", type=int, default=50)   # 最大训练轮数
parser.add_argument("--update_encoder_epochs", type=int, default=8) # 编码器更新的轮数
parser.add_argument("--devices", type=int, default=1)       # 使用的GPU数量
parser.add_argument("--max_length", type=int, default=1024) # 序列最大长度
parser.add_argument("--accumulate_grad_batches", type=int, default=4) # 梯度累积的批次数
parser.add_argument("--first_nrows", type=int, default=-1)  # 用于调试时限制数据量
# 选择要使用的模型
parser.add_argument("--model_name", type=str, default="ctpd",
                    choices=["proto_ts", "ipnet", "grud", "seft", "mtand", "dgm2",
                             "medfuse", "cnn", "utde", "ctpd", "copula", "lstm"])
parser.add_argument("--ts_learning_rate", type=float, default=4e-5) # 时间序列模型的学习率
parser.add_argument("--ckpt_path", type=str, default="")    # 模型检查点路径
parser.add_argument("--test_only", action="store_true")     # 是否只进行测试
# 选择池化类型
parser.add_argument("--pooling_type", type=str, default="mean",
                    choices=["attention", "mean", "last"])
parser.add_argument("--use_prototype", action="store_true") # 是否使用原型学习
parser.add_argument("--use_multiscale", action="store_true") # 是否使用多尺度特征
# 损失函数的权重参数
parser.add_argument("--lamb1", type=float, default=0.5)
parser.add_argument("--lamb2", type=float, default=0)
parser.add_argument("--lamb3", type=float, default=0)
parser.add_argument("--num_slots", type=int, default=16)    # 原型槽的数量
parser.add_argument("--lamb_copula", type=float, default=0.00001) # copula损失权重
args = parser.parse_args()

'''
运行示例命令：
CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --devices 1 --task ihm --batch_size 128 --model_name utde 
CUDA_VISIBLE_DEVICES=4 python train_mimic3.py --devices 1 --task pheno --batch_size 128 --model_name utde 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --devices 1 --task los --batch_size 128 --model_name ctpd
'''

# 设置时间序列特征的维度
args.orig_reg_d_ts = 34  # 原始规范化时间序列特征维度
args.orig_d_ts = 17      # 原始时间序列特征维度

def cli_main():
    # 用于存储不同随机种子下的评估指标
    all_auroc = []  # 曲线下面积
    all_auprc = []  # 精确率-召回率曲线下面积
    all_f1 = []     # F1分数

    # 用于回归任务
    all_mse = []
    all_r2 = []
    all_mape = []

    # 使用不同的随机种子进行多次实验
    for seed in [41, 42, 43]:
        # ----------------(1) 根据任务类型设置参数----------------
        seed_everything(seed)  # 设置随机种子，确保实验可重复性

        # 定义数据模块
        if args.first_nrows == -1:
            args.first_nrows = None

        # 根据任务类型设置相应的参数
        if args.task in ["ihm", "readm"]:
            args.period_length = 48  # 只看前48小时来预测in-hospital mortality
            args.num_labels = 2      # 二分类任务
        elif args.task == "pheno":
            args.period_length = 24  # 只看前24小时来预测phenotype
            args.num_labels = 25     # 多分类任务
        elif args.task == "los":
            args.period_length = 24   # 只看前24小时来预测剩余住院时长
            args.num_labels = 1      # 回归：只有1个连续值输出
        else:
            raise ValueError("Unsupported task")

        # 初始化数据模块
        dm = MIMIC3DataModule(
            file_path=str(MIMIC3_IHM_PATH),
            tt_max=args.period_length,
            batch_size=args.batch_size,
            modeltype="TS_Text",
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)

        # 定义模型
        if args.test_only:
            args.devices = 1  # 测试时只使用一个设备

        # 根据模型名称选择并初始化模型
        if args.model_name == "ipnet":
            if args.ckpt_path:
                model = IPNetModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = IPNetModule(**vars(args))
        elif args.model_name == "proto_ts":
            if args.ckpt_path:
                model = ProtoTSModel.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = ProtoTSModel(**vars(args))
        elif args.model_name == "grud":
            if args.ckpt_path:
                model = GRUDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = GRUDModule(**vars(args))
        elif args.model_name == "seft":
            if args.ckpt_path:
                model = SEFTModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = SEFTModule(**vars(args))
        elif args.model_name == "mtand":
            if args.ckpt_path:
                model = MTANDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MTANDModule(**vars(args))
        elif args.model_name == "dgm2":
            if args.ckpt_path:
                model = DGM2OModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = DGM2OModule(**vars(args))
        elif args.model_name == "medfuse":
            if args.ckpt_path:
                model = MedFuseModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = MedFuseModule(**vars(args))
        elif args.model_name == "cnn":
            if args.ckpt_path:
                model = CNNModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CNNModule(**vars(args))
        elif args.model_name == "lstm":
            if args.ckpt_path:
                model = LSTMModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = LSTMModule(**vars(args))
        elif args.model_name == "utde":
            if args.ckpt_path:
                model = UTDEModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = UTDEModule(**vars(args))
        elif args.model_name == "ctpd":
            if args.ckpt_path:
                model = CTPDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CTPDModule(**vars(args))
        elif args.model_name == "copula":
            if args.ckpt_path:
                model = CopulaModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
            else:
                model = CopulaModule(**vars(args))

        else:
            raise ValueError("Invalid model name")  # 如果模型名称无效，抛出错误

        # 计算每个epoch的训练迭代次数
        model.train_iters_per_epoch = len(dm.train_dataloader()) // (args.accumulate_grad_batches * args.devices)

        # 初始化训练器
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"mimic3_{args.task}_{args.model_name}_{run_name}_demo_test"
        os.makedirs(ROOT_PATH / "log/ckpts", exist_ok=True)
        # 设置wandb日志记录器
        logger = WandbLogger(
            name=run_name,
            save_dir=str(ROOT_PATH / "log"),
            project="copula", log_model=False)

        # ----------------(2) 为不同任务设置回调----------------
        """
        停止条件:EarlyStopping
        监控 验证集 AUPRC (val_auprc) 或 AUROC (val_auroc)，取决于任务类型：
        - "ihm"、"readm" 任务 监控 "val_auprc"
        - "pheno" 任务 监控 "val_auroc"
        如果连续 5 轮(patience=5)没有提升，则停止训练。

        保存最佳模型:ModelCheckpoint
        根据 val_auprc(或 val_auroc) 的最大值，保存最佳模型。
        save_top_k=2:保存 两个最好的检查点（如果 val_auprc 继续提升，则不断覆盖最差的一个）。
        save_last=False:不会保存最后一轮的模型，只保留最好的两个。

        在 PyTorch Lightning 中,configure_optimizers()与ModelCheckpoint之间的关系主要体现在:
        - configure_optimizers() 定义了训练的优化策略(如优化器和学习率调整策略)。
        - ModelCheckpoint负责在训练过程中根据某个监控指标(monitor)自动保存模型checkpoint。

        """
        if args.task in ["ihm", "readm"]:
            callbacks = [
                LearningRateMonitor(logging_interval="step"),  # 监控学习率
                ModelCheckpoint(                               # 模型检查点保存
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_auprc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auprc", patience=5,  # 早停策略
                              mode="max", verbose=True)
            ]
        
        elif args.task == "pheno":
            # pheno任务使用auroc作为监控指标
            callbacks = [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_auroc",
                    mode="max",
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_auroc", patience=5,
                              mode="max", verbose=True)
            ]
        elif args.task == "los":
            # 例：使用 R² 来监控并保存最佳模型
            callbacks = [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    dirpath=str(ROOT_PATH / "log/ckpts" / run_name),
                    monitor="val_r2",
                    mode="max",  # R² 越大越好
                    save_top_k=2,
                    save_last=False),
                EarlyStopping(monitor="val_r2", patience=5, mode="max", verbose=True)
            ]
        # 配置训练器
        trainer = Trainer(
            devices=args.devices,                    # GPU数量
            accelerator="gpu",                       # 使用GPU加速
            max_epochs=args.max_epochs,              # 最大训练轮数
            precision="16-mixed",                    # 使用混合精度训练
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=callbacks,                     # 设置回调函数
            logger=logger,                          # 设置日志记录器
            strategy="auto",  # 分布式训练策略
            gradient_clip_val=0.5,                  # 梯度裁剪值
        )

        # 训练和测试模型
        if not args.test_only:
            trainer.fit(model, dm)                  # 训练模型
            trainer.test(model, datamodule=dm, ckpt_path="best")  # 使用最佳检查点测试
        else:
            trainer.test(model, datamodule=dm)      # 仅进行测试
        
        # ---------------(3) 收集并保存评估指标---------------
        # 不同任务，模型里保存/输出的指标字段可能不一样
        if args.task in ["ihm", "readm", "pheno"]:
            all_auroc.append(model.report_auroc)
            all_auprc.append(model.report_auprc)
            all_f1.append(model.report_f1)
        elif args.task == "los":
            # 收集LOS任务的评估指标
            all_mse.append(model.report_mse)
            all_r2.append(model.report_r2)
            all_mape.append(model.report_mape)

        wandb.finish()  # 结束wandb日志记录

    # ---------------(4) 计算统计结果---------------
    if args.task in ["ihm", "readm", "pheno"]:
        report_df = pd.DataFrame({
            "auroc": all_auroc,
            "auprc": all_auprc,
            "f1": all_f1
        })
        mean_df = report_df.mean(axis=0)
        std_df = report_df.std(axis=0)
        statistic_df = pd.concat([mean_df, std_df], axis=1)
        statistic_df.columns = ["mean", "std"]
        print(statistic_df)
    elif args.task == "los":
        # 对 los 任务，输出 MSE/R^2/MAPE
        report_df = pd.DataFrame({
            "mse": all_mse,
            "r2": all_r2,
            "mape": all_mape
        })
        mean_df = report_df.mean(axis=0)
        std_df = report_df.std(axis=0)
        statistic_df = pd.concat([mean_df, std_df], axis=1)
        statistic_df.columns = ["mean", "std"]
        print("住院时长预测统计指标：")
        print(statistic_df)


if __name__ == "__main__":
    cli_main()
