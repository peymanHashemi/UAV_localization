# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
# from models.model import make_model
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from tool.utils import save_network, copyfiles2checkpoints, get_preds, get_logger, calc_flops_params, set_seed
import warnings
from losses.loss import Loss
import matplotlib.pyplot as plt
import os
import subprocess
import sys
from val import val
import json


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--checkpoint', default='net.pth',
                    type=str, help='save model path')
    parser.add_argument('--name', default='test',
                        type=str, help='the experiment name that will be saved in checkpoints dir in the root')
    parser.add_argument('--data_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/train',
                        type=str, help='training dir path')
    parser.add_argument('--val_dir', default="/home/abbas/AUT/Dataset/DenseUAV/google2/validation", type=str, help='head pooling type for applying')
    parser.add_argument('--seg_dir', default="/home/abbas/AUT/Dataset/DenseUAV/google2/segment", type=str, help='head pooling type for applying')
    parser.add_argument('--ms', default='1', type=str,
                        help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    parser.add_argument('--mode',default='1', type=int,help='1:drone->satellite   2:satellite->drone')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--rr', default="", type=str, help='random rotate')
    parser.add_argument('--ra', default="", type=str, help='random affine')
    parser.add_argument('--re', default="", type=str, help='random erasing')
    parser.add_argument('--cj', default="", type=str, help='color jitter')
    parser.add_argument('--erasing_p', default=0.3, type=float,
                        help='random erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--DA', action='store_true',
                        help='use Color Data Augmentation')
    parser.add_argument('--droprate', default=0.5,
                        type=float, help='drop rate')
    parser.add_argument('--autocast', action='store_true',
                        default=True, help='use mix precision')
    parser.add_argument('--block', default=2, type=int, help='')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--feature_loss', default="no", type=str, help='loss type of metric learning')
    parser.add_argument('--kl_loss', default="no", type=str, help='loss type of mutual learning')
    parser.add_argument('--sample_num', default=1, type=int,
                        help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=120, type=int, help='total epoches for training')
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--load_from', default="", type=str, help='checkpoints path for pre-loading')
    parser.add_argument('--backbone', default="cvt13", type=str, help='backbone network for applying')
    parser.add_argument('--head', default="FSRA_CNN", type=str, help='head type for applying')
    parser.add_argument('--head_pool', default="max", type=str, help='head pooling type for applying')
    parser.add_argument('--lr_step1', default=70, type=int, help='head pooling type for applying')
    parser.add_argument('--lr_step2', default=110, type=int, help='head pooling type for applying')
    parser.add_argument('--lr_step3', default=1000, type=int, help='head pooling type for applying')
    parser.add_argument('--lr_step4', default=10000, type=int, help='head pooling type for applying')
    parser.add_argument('--val_step', default=5, type=int, help='head pooling type for applying')

    parser.add_argument('--segmentaion', default=False, help='head pooling type for applying')
    parser.add_argument('--seg_backbone', default="cvt13", type=str, help='backbone network for applying')
    parser.add_argument('--seg_val_dir', default="/home/abbas/AUT/Dataset/DenseUAV/google2/seg_val", type=str, help='head pooling type for applying')

    
    opt = parser.parse_args()
    print(opt)
    return opt

def plot_curves(total_epoch_loss, total_epoch_cls_loss, total_epoch_kl_loss, total_epoch_triplet_loss, total_epoch_acc, total_epoch_acc2, Recall_1, Recall_5):  
    plot_dir = f'./checkpoints/{opt.name}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(total_epoch_triplet_loss, label='Triplet_Loss')
    plt.legend()
    plt.title(f"Triplet_Loss BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
  #save fig in checkpoints
    plt.savefig(f"checkpoints/{opt.name}/plots/Triplet_Loss{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")

    plt.figure()
    plt.plot(total_epoch_kl_loss, label='KL_Loss')
    plt.legend()
    plt.title(f"KL_Loss BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
  #save fig in checkpoints
    plt.savefig(f"checkpoints/{opt.name}/plots/KL_Loss{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")

    plt.figure()
    plt.plot(total_epoch_loss, label='Loss')
    plt.plot(total_epoch_cls_loss, label='Cls_Loss')
    plt.legend()
    plt.title(f"loss1 BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(f"checkpoints/{opt.name}/plots/loss1_{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")
    
    plt.figure()
    plt.plot(total_epoch_acc, label='Satellite_Acc')
    plt.plot(total_epoch_acc2, label='Drone_Acc')
    plt.legend()
    plt.title(f"acc BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('acc')   
    plt.show()
  #save fig in checkpoints
    plt.savefig(f"checkpoints/{opt.name}/plots/acc_{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")

    ############################################################################
    #plotting val
    #R@1
    plt.figure()
    plt.plot(Recall_1, label='Recall@1')
    plt.legend()
    plt.title(f"Recall@1 BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
  #save fig in checkpoints
    plt.savefig(f"checkpoints/{opt.name}/plots/Recall@1_{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")
    #R@5
    plt.figure()
    plt.plot(Recall_5, label='Recall@5')
    plt.legend()
    plt.title(f"Recall@5 BB: {opt.backbone} - lr: {opt.lr} - epoch: {opt.num_epochs}")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
  #save fig in checkpoints
    plt.savefig(f"checkpoints/{opt.name}/plots/Recall@5_{opt.backbone}_{opt.lr}_{opt.num_epochs}.png")


def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes):
    logger = get_logger(
        "checkpoints/{}/train.log".format(opt.name))

    # thop计算MACs
    # macs, params = calc_flops_params(
    #     model, (1, 3, opt.h, opt.w), (1, 3, opt.h, opt.w))
    # logger.info("model MACs={}, Params={}".format(macs, params))

#py_filepath = 'C:/Users/benb/Desktop/flaskEconServer/plots.py'
    # val_path = 'val.py'

    # val_run = '"%s" "%s" "%s"' % (sys.executable,                  # command
    #                         val_path,                     # argv[0]
    #                         os.path.basename(val_path))   # argv[1]

    

    model_testing_name = f"\nname of test : {opt.name}\n-> backbone: {opt.backbone} - lr: {opt.lr} - batchsize: {opt.batchsize} - epoch: {opt.num_epochs} - lr_step1: {opt.lr_step1} - lr_step2: {opt.lr_step2}- head: {opt.head} - head_pool: {opt.head_pool} - feature_loss: {opt.feature_loss}"
    with open(f"all_results.txt", "a") as F:
        F.write(model_testing_name)

    #saving losses and accuracy 
    total_epoch_loss = []
    total_epoch_cls_loss = []
    total_epoch_kl_loss = []
    total_epoch_triplet_loss = []
    total_epoch_acc = []
    total_epoch_acc2 = []
    total_satellite_acc = []
    total_drone_acc = []
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()
    scaler = GradScaler()
    nnloss = Loss(opt)
    recall_1= []
    recall_5= []
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 50)

        model.train(True)  # Set model to training mode
        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        for data, data3, data5, data7 in dataloaders:
            # 获取输入无人机和卫星数据
            inputs, labels = data
            inputs3, labels3 = data3
            inputs5, labels5 = data5
            inputs7, labels7 = data7

            now_batch_size = inputs.shape[0]
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
                inputs3 = Variable(inputs3.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels3 = Variable(labels3.cuda().detach())

                inputs5 = Variable(inputs5.cuda().detach())
                inputs7 = Variable(inputs7.cuda().detach())
                labels5 = Variable(labels5.cuda().detach())
                labels7 = Variable(labels7.cuda().detach())

            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # start_time = time.time()
            # 模型前向传播
            with autocast():
                outputs, outputs2 = model(inputs, inputs3, inputs5, inputs7)
            # print("model_time:{}".format(time.time()-start_time))
            # 计算损失
            loss, cls_loss, f_triplet_loss, kl_loss = nnloss(
                outputs, outputs2, labels, labels3)
            # start_time = time.time()
            # 反向传播
            if opt.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # print("backward_time:{}".format(time.time()-start_time))

            # 统计损失
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item()*now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
            running_kl_loss += kl_loss.item() * now_batch_size

            # 统计精度
            preds, preds2 = get_preds(outputs[0], outputs2[0])
            if isinstance(preds, list) and isinstance(preds2, list):
                running_corrects += sum([float(torch.sum(pred == labels.data))
                                        for pred in preds])/len(preds)
                running_corrects2 += sum([float(torch.sum(pred == labels3.data))
                                         for pred in preds2]) / len(preds2)
            else:
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels3.data))
            # for i in range(10):
            # plt.imshow(inputs.numpy()[0])
            

        # 统计损失和精度
        epoch_cls_loss = running_cls_loss/dataset_sizes['satellite']
        epoch_kl_loss = running_kl_loss / dataset_sizes['satellite']
        epoch_triplet_loss = running_triplet/dataset_sizes['satellite']
        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_acc = running_corrects / dataset_sizes['satellite']
        epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
        lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        logger.info('Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                    .format(epoch_loss, epoch_cls_loss, epoch_kl_loss,
                            epoch_triplet_loss, epoch_acc,
                            epoch_acc2, lr_backbone, lr_other))
        total_epoch_loss.append(epoch_loss)
        total_epoch_cls_loss.append(epoch_cls_loss)
        total_epoch_kl_loss.append(epoch_kl_loss)
        total_epoch_triplet_loss.append(epoch_triplet_loss)
        total_epoch_acc.append(epoch_acc)
        total_epoch_acc2.append(epoch_acc2)
        total_satellite_acc.append(epoch_acc)
        total_drone_acc.append(epoch_acc2)
        scheduler.step()
        save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        since = time.time()
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # subprocess.run(val_run)
        if epoch % opt.val_step == 0 or epoch == (opt.num_epochs -1):
            os.system(f"python3 val2.py --name {opt.name} --val_dir {opt.val_dir} --gpu_ids {opt.gpu_ids} --num_worker {opt.num_worker} --seg_val_dir {opt.seg_val_dir}" )

        # recall_1, recall_5 = val(opt,recall_1, recall_5)
    # recall_1 = [float(t.item()) for t in recall_1]
    # recall_5= [float(t.item()) for t in recall_5]
    with open(f"checkpoints/{opt.name}/recall@1.json", "r") as infile:
        recall_1 = json.load(infile)
    with open(f"checkpoints/{opt.name}/recall@5.json", "r") as infile:
        recall_5 = json.load(infile)
    plot_curves(total_epoch_loss, total_epoch_cls_loss, total_epoch_kl_loss, total_epoch_triplet_loss, total_epoch_acc, total_epoch_acc2,recall_1,recall_5)
    
    info = 'backbone:%s learning rate:%f bachsize:%f num_epochs:%f \nlr_step1:%f lr_step2:%f lr_step3:%f lr_step4:%f \nmetric learning:%s representation learning:%s mutual learning:%s \nhead:%s head_pool:%s random rotate:%s random affine:%s random erasing:%s collor jitter:%s'%(opt.backbone,opt.lr,opt.batchsize,opt.num_epochs,opt.lr_step1,opt.lr_step2,opt.lr_step3,opt.lr_step4,opt.cls_loss,opt.feature_loss, opt.kl_loss, opt.head, opt.head_pool,opt.rr,opt.ra,opt.re,opt.cj)
    with open(f"checkpoints/{opt.name}/information.txt", "w") as F:
        F.write(info)

    
    
    
    

    # with open(f"checkpoints/{opt.name}/recall@1.json", 'w') as F:
    #     json.dump(recall_1, F, indent=4)
    # with open(f"checkpoints/{opt.name}/recall@5.json", 'w') as F:
    #     json.dump(recall_5, F, indent=4)
    # with open(f"checkpoints/{opt.name}/recall@1.json", "w") as F:
    #     F.write(recall_1)
    # with open(f"checkpoints/{opt.name}/recall@5.json", "w") as F:
    #     F.write(recall_5)




if __name__ == '__main__':
    set_seed(666)

    opt = get_parse()
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)
    copyfiles2checkpoints(opt)
    model = make_model(opt)

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)

    if use_gpu:
        model = model.cuda()
    # 移动文件到指定文件夹
    
    with open(f"checkpoints/{opt.name}/recall@1.json", 'w') as F:
        json.dump([], F, indent=4)
    with open(f"checkpoints/{opt.name}/recall@5.json", 'w') as F:
        json.dump([], F, indent=4)
    train_model(model, opt, optimizer_ft, exp_lr_scheduler,
                dataloaders, dataset_sizes)
