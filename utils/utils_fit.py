import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
import numpy as np


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    torch.autograd.set_detect_anomaly(True)
    loss        = 0
    val_loss    = 0

    # 每个epoch只跑1/5的batch
    epoch_step = epoch_step // 5 
    
    # 展示loss和学习率
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    # 训练模式
    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        # 遍历训练数据 得到图片、目标框、语言描述、多目标框和运动关系
        images, targets, captions, multi_targets, relation = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            # 数据转到GPU
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                captions = torch.tensor(np.array(captions))
                captions  = captions.cuda(local_rank)
                relation = torch.tensor(np.array(relation))
                relation = relation.cuda(local_rank)

                for target in multi_targets:
                    target = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in target]
                    target = [ann.cuda(local_rank) for ann in target]

        # batch梯度清零
        optimizer.zero_grad()
        if not fp16:   
            # 前向传播 loss
            outputs, motion_loss = model_train(images, captions, multi_targets, relation)
            loss_value = yolo_loss(outputs, targets) + motion_loss

            # 反向传播 参数更新
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images) 
                loss_value = yolo_loss(outputs, targets)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            # ema更新
            ema.update(model_train)

        # 展示平均loss和当前学习率
        loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
    
    # 训练结束 验证开始
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    # 使用历史平滑权重参数
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
    
    # 验证模式
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            # 计算loss
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    # 验证结束 记录日志 调用回调函数计算评价
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # 保存ema参数
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # 保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        # 保存最佳模型
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        
        # 保存last模型
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
