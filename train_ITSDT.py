import datetime
import os,random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pycocotools.coco as coco
from nets.MoPKL import MoPKL
from nets.training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset
from utils.dataloader_for_ITSDT import seqDataset, dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    
    Cuda            = True
    distributed     = False # 使用分布式训练
    sync_bn         = False # 使用同步批归一化
    fp16            = False # 使用混合精度训练
    classes_path    = 'D:/Github/MyMoPKL/MoPKL-main/model_data/classes.txt' # 类别路径
    model_path      = 'D:/Github/MyMoPKL/MoPKL-main/model_data/pre_trained_backbone.pth' # 预训练模型路径
    input_shape     = [512, 512] # 输入图像大小
    phi             = 's' # 模型大小
    mosaic              = False # mosaic数据增强
    mosaic_prob         = 0.5
    mixup               = False # mixup数据增强
    mixup_prob          = 0.5 # mixup概率
    special_aug_ratio   = 0.7 # 特殊数据增强比例
    Init_Epoch          = 0 # 初始训练轮数
    Freeze_Epoch        = 0 # 冻结训练轮数
    Freeze_batch_size   = 4
    UnFreeze_Epoch      = 100 # 解冻训练轮数
    Unfreeze_batch_size = 4 
    Freeze_Train        = False # 是否冻结训练
    Init_lr             = 1e-2 # 初始学习率
    Min_lr              = Init_lr * 0.01 # 最小学习率
    optimizer_type      = "sgd" # 优化器类型 随机梯度下降
    momentum            = 0.937 # 动量系数 用于加速梯度下降
    weight_decay        = 5e-4 # L2正则化系数
    lr_decay_type       = "cos" # 余弦退火学习率衰减
    save_period         = 1 # 权重保存周期
    save_dir            = 'D:/Github/MyMoPKL/logs' # 权重保存路径
    eval_flag           = True # 模型验证
    eval_period         = 100
    num_workers         = 1
    train_annotation_path = 'D:/Github/ITSDT/coco_train_ITSDT.txt' # 训练数据集路径
    val_annotation_path = 'D:/Github/ITSDT/coco_val_ITSDT.txt' # 验证数据集路径
    
    ngpus_per_node  = torch.cuda.device_count() # gpu数
    
    # 是否启用分布式训练
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        # 单卡训练
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
    
    # 设置随机种子
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 获取类别信息 target 1
    class_names, num_classes = get_classes(classes_path)
    
    # 初始化模型 1个类别 2帧输入
    model = MoPKL(num_classes=1,  num_frame=2) 
    weights_init(model) 

    # 加载预训练权重
    if model_path != '':
        
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
       
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，部分参数没有载入是正常现象，这里只使用了预训练模型的部分参数权重。\033[0m")

    # 创建yolo损失实例
    yolo_loss    = YOLOLoss(num_classes, fp16, strides=[8])
   
    # 创建日志
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
    
    # 是否使用混合精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # 训练模式
    model_train     = model.train()
    
    # 是否同步批归一化
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # 是否使用cuda和分布式训练
    if Cuda:
        if distributed:
            
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 指数滑动平均
    ema = ModelEMA(model_train)
    
    # 统计训练集验证集样本量
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    # 打印训练配置
    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = log_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small for training. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d. \033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data amount of this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d. \033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total epoch to %d. \033[0m"%(total_step, wanted_step, wanted_epoch))

    
    if True:
        # 是否冻结训练
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 定义批大小      
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 定义学习率
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # 定义优化器参数 是否加入weight decay (L2正则化)
        pg0, pg1, pg2 = [], [], []  
        
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   

        # 定义优化器 添加参数组    
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        
        # 构建学习率衰减函数
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        # 计算step 即batch数
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset. ")
        
        # ema更新
        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        # 构建训练/验证集
        train_dataset = seqDataset(train_annotation_path, input_shape[0], 2, 'train')
        val_dataset = seqDataset(val_annotation_path, input_shape[0], 2, 'val')

        # 是否数据打乱
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True 

        # 构建训练/验证集DataLoader
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)
        # 模型评估
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            
            # 是否解冻
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                    
                nbs             = 64 # 基准batch size
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max) # 根据实际batch size线性调整学习率
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
               
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch) # 学习率衰减函数
                
                # 解冻backbone
                for param in model.backbone.parameters():
                    param.requires_grad = True
        
                epoch_step      = num_train // batch_size 
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            # 更新epoch
            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)
            
            # 更新学习率
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 开始一个epoch训练
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, log_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()


