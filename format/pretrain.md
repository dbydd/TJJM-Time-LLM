```python

# %%

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler

from time_llm.data_provider_pretrain.data_factory import data_provider
from time_llm.models import Autoformer, DLinear, TimeLLM

import time
import random
import numpy as np
import os

from time_llm.utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

# 导入必要的库和模块
# accelerate: 用于分布式训练的库
# torch: PyTorch深度学习框架
# nn, optim, lr_scheduler: PyTorch中的神经网络, 优化器和学习率调度器模块
# time_llm.data_provider_pretrain.data_factory: Time-LLM项目中的数据提供器
# time_llm.models: Time-LLM项目中的模型(Autoformer, DLinear, TimeLLM)
# time, random, numpy, os: Python标准库中的时间, 随机数, 数值计算和操作系统相关模块
# time_llm.utils.tools: Time-LLM项目中的工具函数

# %%

# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 设置环境变量(这两行被注释掉了)
# CURL_CA_BUNDLE: 设置CA证书路径
# PYTORCH_CUDA_ALLOC_CONF: 设置PyTorch的CUDA内存分配策略

parser = argparse.ArgumentParser(description='Time-LLM')

# 创建一个命令行参数解析器,用于解析命令行参数

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 设置随机数种子,以保证实验的可重复性
# 分别为Python的random模块, PyTorch和NumPy设置随机数种子

# %%
args = {}

# 创建一个空字典,用于存储命令行参数和默认参数

# basic config
args.task_name = 'long_term_forecast' # 'task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]'
args.is_training = 1 # 'status'
args.model_id = 'test' # 'model id'
args.model_comment = 'none' # 'prefix when saving test results'
args.model = 'Autoformer' # 'model name, options: [Autoformer, DLinear]'
args.seed = 2021 # 'random seed'

# 设置基本配置参数
# task_name: 任务名称,可选项包括长期预测, 短期预测, 插值, 分类, 异常检测
# is_training: 训练状态
# model_id: 模型ID
# model_comment: 保存测试结果时的前缀
# model: 模型名称,可选项包括Autoformer和DLinear
# seed: 随机数种子

# data loader
args.data_pretrain = 'ETTm1' # 'dataset type'
args.data = 'ETTm1' # 'dataset type'
args.root_path = 'ETTh1' # 'data file'
args.data_path_pretrain = 'ETTh1' # 'data file'
args.features = 'M' # 'forecasting task, options:[M, S, MS]; ''M:multivariate predict multivariate, S: univariate predict univariate, ''MS:multivariate predict univariate'
args.target = 'OT' # 'target feature in S or MS task'
args.loader = 'modal' # 'dataset type'
args.freq = 'h' # 'freq for time features encoding, ''options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], ''you can also use more detailed freq like 15min or 3h'
args.checkpoints = 96 # 'input sequence length'
args.label_len = 48 # 'start token length'
args.pred_len = 96 # 'prediction sequence length'
args.seasonal_patterns = 'Monthly' # 'subset for M4'

# 设置数据加载器相关参数
# data_pretrain: 预训练数据集类型
# data: 数据集类型
# root_path: 数据文件路径
# data_path_pretrain: 预训练数据文件路径
# features: 预测任务类型,可选项包括M(多变量预测多变量), S(单变量预测单变量), MS(多变量预测单变量)
# target: S或MS任务中的目标特征
# loader: 数据集类型
# freq: 时间特征编码的频率,可选项包括s(秒), t(分), h(小时), d(天), b(工作日), w(周), m(月),也可以使用更详细的频率如15min或3h
# checkpoints: 输入序列长度
# label_len: 开始令牌长度
# pred_len: 预测序列长度
# seasonal_patterns: M4数据集的子集

# model define
args.enc_in = 7 # 'encoder input size'
args.dec_in = 7 # 'decoder input size'
args.c_out = 7 # 'output size'
args.d_model = 16 # 'dimension of model'
args.n_heads = 8 # 'num of heads'
args.e_layers = 2 # 'num of encoder layers'
args.d_layers = 1 # 'num of decoder layers'
args.d_ff = 32 # 'dimension of fcn'
args.moving_avg = 25 # 'window size of moving average'
args.factor = 1 # 'attn factor'
args.dropout = 0 # 'dropout'
args.embed = 'timeF' # 'time features encoding, options:[timeF, fixed, learned]'
args.activation = 'gelu' # 'activation'
args.output_attention = 16 # 'patch length'
args.stride = 8 # 'stride'
args.prompt_domain = 0 # ''
args.llm_model = 'LLAMA' # 'LLM model' # LLAMA, GPT2, BERT
args.llm_dim = '4096' # 'LLM model dimension'# LLama7b:4096; GPT2-small:768; BERT-base:768

# 设置模型相关参数
# enc_in: 编码器输入大小
# dec_in: 解码器输入大小
# c_out: 输出大小
# d_model: 模型维度
# n_heads: 注意力头数
# e_layers: 编码器层数
# d_layers: 解码器层数
# d_ff: 前馈网络维度
# moving_avg: 移动平均窗口大小
# factor: 注意力因子
# dropout: dropout比率
# embed: 时间特征编码方式,可选项包括timeF(时间特征), fixed(固定), learned(学习)
# activation: 激活函数
# output_attention: 输出注意力的patch长度
# stride: 步长
# prompt_domain: 提示域
# llm_model: 大语言模型,可选项包括LLAMA, GPT2, BERT
# llm_dim: 大语言模型维度,LLama7b为4096,GPT2-small为768,BERT-base为768

# optimization
args.num_workers = 10 # 'data loader num workers'
args.itr = 1 # 'experiments times'
args.train_epochs = 10 # 'train epochs'
args.align_epochs = 10 # 'alignment epochs'
args.batch_size = 32 # 'batch size of train input data'
args.eval_batch_size = 8 # 'batch size of model evaluation'
args.patience = 5 # 'early stopping patience'
args.learning_rate = 0 # 'optimizer learning rate'
args.des = 'test' # 'exp description'
args.loss = 'MSE' # 'loss function'
args.lradj = 'type1' # 'adjust learning rate'
args.pct_start = 0 # 'pct_start'
args.use_amp = False # use automatic mixed precision training
args.llm_layers = False
args.percent = 100

# 设置优化相关参数
# num_workers: 数据加载器的工作进程数
# itr: 实验次数
# train_epochs: 训练轮数
# align_epochs: 对齐轮数
# batch_size: 训练输入数据的批大小
# eval_batch_size: 模型评估的批大小
# patience: 早停的耐心值
# learning_rate: 优化器学习率
# des: 实验描述
# loss: 损失函数
# lradj: 学习率调整方式
# pct_start: 学习率调整的起始百分比
# use_amp: 是否使用自动混合精度训练
# llm_layers: 大语言模型层数
# percent: 百分比

# %%

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# 设置分布式训练相关参数
# ddp_kwargs: 分布式数据并行的关键字参数,设置find_unused_parameters为True
# deepspeed_plugin: DeepSpeed插件,用于加速训练(这里被注释掉了)
# accelerator: Accelerator对象,用于加速训练,传入ddp_kwargs作为参数

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    # 循环进行多次实验
    # 设置实验记录,包括任务名称,模型ID,模型名称,数据集,特征,序列长度,标签长度,预测长度,模型维度,注意力头数,编码器层数,解码器层数,前馈网络维度,因子,嵌入方式,实验描述和实验次数

    train_data, train_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'train')
    vali_data, vali_loader = data_provider(args, args.data_pretrain, args.data_path_pretrain, True, 'val')
    test_data, test_loader = data_provider(args, args.data, args.data_path, False, 'test')

    # 加载训练集,验证集和测试集数据
    # train_data: 训练数据
    # train_loader: 训练数据加载器
    # vali_data: 验证数据
    # vali_loader: 验证数据加载器
    # test_data: 测试数据
    # test_loader: 测试数据加载器

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    # 根据模型名称创建模型对象
    # 如果模型名称为Autoformer,则创建Autoformer模型
    # 如果模型名称为DLinear,则创建DLinear模型
    # 否则创建TimeLLM模型

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    # 设置检查点保存路径
    # path: 唯一的检查点保存路径,由设置和模型注释组成
    # args.content: 加载内容
    # 如果路径不存在且当前进程是主进程,则创建路径

    time_now = time.time()

    # 记录当前时间

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # 设置训练步数和早停对象
    # train_steps: 训练步数,等于训练数据加载器的长度
    # early_stopping: 早停对象,传入accelerator和耐心值

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    # 获取需要训练的参数
    # trained_parameters: 需要训练的参数列表,包括模型中requires_grad为True的参数

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # 创建优化器
    # model_optim: Adam优化器,传入需要训练的参数和学习率

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

# 如果学习率调整方式不是COS,则创建OneCycleLR调度器
# scheduler: OneCycleLR调度器,传入优化器,每个epoch的步数,起始百分比,训练轮数和最大学习率

criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

# 创建损失函数和评估指标
# criterion: MSE损失函数
# mae_metric: MAE损失函数

train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
    train_loader, vali_loader, test_loader, model, model_optim, scheduler)

# 使用accelerator准备数据加载器,模型,优化器和调度器,以便在分布式环境中使用

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

# 如果使用自动混合精度训练,则创建GradScaler对象

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()

    # 开始训练循环
    # iter_count: 迭代计数器
    # train_loss: 训练损失列表
    # model.train(): 将模型设置为训练模式
    # epoch_time: 当前epoch的开始时间

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)

        # 遍历训练数据加载器
        # iter_count: 迭代计数器加1
        # model_optim.zero_grad(): 清零梯度
        # 将输入数据移动到accelerator的设备上并转换为float类型

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
            accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
            accelerator.device)

        # 创建解码器输入
        # dec_inp: 解码器输入,由标签和全零张量拼接而成

        # encoder - decoder
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        else:
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

        # 编码器-解码器前向传播
        # 如果使用自动混合精度训练,则使用autocast上下文
        # 如果输出注意力,则取模型输出的第一个元素作为输出
        # f_dim: 根据特征类型设置输出的维度
        # 取输出和标签的最后pred_len个时间步和特定维度
        # 计算损失并将其添加到训练损失列表中

        if (i + 1) % 100 == 0:
            accelerator.print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        # 每100次迭代打印训练信息
        # 打印当前迭代次数,epoch,损失值
        # 计算训练速度和剩余时间并打印
        # 重置迭代计数器和当前时间

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            accelerator.backward(loss)
            model_optim.step()

        # 反向传播和优化
        # 如果使用自动混合精度训练,则使用scaler进行梯度缩放,优化器步进和scaler更新
        # 否则使用accelerator的backward方法进行反向传播,并进行优化器步进

        if args.lradj == 'TST':
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

    # 如果学习率调整方式为TST,则调整学习率并进行调度器步进

    accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
    test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    accelerator.print(
        "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

    # 打印epoch信息
    # 打印当前epoch耗时
    # 计算平均训练损失
    # 在验证集和测试集上进行评估,获取损失和MAE损失
    # 打印训练损失,验证损失,测试损失和MAE损失

    early_stopping(vali_loss, model, path)
    if early_stopping.early_stop:
        accelerator.print("Early stopping")
        break

    # 进行早停判断
    # 如果触发早停,则打印早停信息并跳出训练循环

    if args.lradj != 'TST':
        if args.lradj == 'COS':
            scheduler.step()
            accelerator.print("lr = {:.10f}".format(model_optim.param_groups['lr']))
        else:
            if epoch == 0:
                args.learning_rate = model_optim.param_groups['lr']
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups['lr']))
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

    else:
        accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()))

# 如果学习率调整方式不是TST
# 如果是COS,则进行调度器步进并打印当前学习率
# 否则,如果是第一个epoch,则将当前学习率赋值给args.learning_rate并打印
# 调整学习率并打印
# 如果是TST,则打印更新后的学习率

accelerator.wait_for_everyone() #等待所有进程完成
if accelerator.is_local_main_process: #如果是本地主进程
path = './checkpoints' # unique checkpoint saving path #删除检查点文件
del_files(path) # delete checkpoint files #打印删除检查点成功的信息
accelerator.print('success delete checkpoints')
#以上就是对剩余代码的注释。这段代码主要包括创建学习率调度器,定义损失函数和评估指标,使用accelerator准备数据加载器、模型、优化器和调度器,进行训练循环,在每个epoch结束后进#行评估和早停判断,根据不同的学习率调整方式进行学习率调整,最后删除检查点文件。
```
