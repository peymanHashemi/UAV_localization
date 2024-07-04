# -*- coding: utf-8 -*-

from __future__ import print_function, division
from datasets.queryDataset import Dataset_query, Query_transforms

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tool.utils import load_network
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1


def extract_feature(opt, model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        
        img, _ = data
        batchsize = img.size()[0]
        count += batchsize
        # if opt.LPN:
        #     # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
        #     ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        # else:
        #     ff = torch.FloatTensor(n, 2048).zero_().cuda()
        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if view_index == 1:
                outputs, _ = model(input_img, None)
            elif view_index == 3:
                _, outputs = model(None, input_img)
            outputs = outputs[1]
            if i == 0:
                ff = outputs
            else:
                ff += outputs
        # norm feature
        if len(ff.shape) == 3:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * \
                np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# evaluation
def evaluate(qf,ql,gf,gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index
    #print(good_index)
    #print(index[0:10])
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
# Load Collected data Trained model
def val(opt, recall_1, recall_5):
    
    opt.h= 256
    opt.w= 256
    opt.ms= '1'
    opt.mode= '1'

    ###load config###
    # load the training config
    # os.system("le")
    config_path = f'checkpoints/{opt.name}/opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    for cfg, value in config.items():
        setattr(opt, cfg, value)

    str_ids = opt.gpu_ids.split(',')
    test_dir = opt.val_dir 

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    str_ms = opt.ms.split(',')
    ms = []

    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True


    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_query_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        # Query_transforms(pad=10,size=opt.w),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_dir = test_dir

    image_datasets_query = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_query_transforms) for x in ['query_drone']}

    image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms) for x in ['gallery_satellite']}

    image_datasets = {**image_datasets_query, **image_datasets_gallery}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=opt.num_worker) for x in ['gallery_satellite', 'query_drone']}
    use_gpu = torch.cuda.is_available()

######################
    opt.checkpoint = f'checkpoints/{opt.name}/net.pth'
    model = load_network(opt)
    # model.classifier.classifier = nn.Sequential()
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    since = time.time()

    if opt.mode == 1:
        query_name = 'query_drone'
        gallery_name = 'gallery_satellite'
    elif opt.mode == 2:
        query_name = 'query_satellite'
        gallery_name = 'gallery_drone'
    else:
        raise Exception("opt.mode is not required")


    which_gallery = which_view(gallery_name)
    which_query = which_view(query_name)

    gallery_path = image_datasets[gallery_name].imgs

    query_path = image_datasets[query_name].imgs

    gallery_label, gallery_path = get_id(gallery_path)
    query_label, query_path = get_id(query_path)

    with torch.no_grad():
        query_feature = extract_feature(
            opt, model, dataloaders[query_name], which_query)
        gallery_feature = extract_feature(
            opt, model, dataloaders[gallery_name], which_gallery)

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
    # scipy.io.savemat('pytorch_result_{}.mat'.format(opt.mode), result)

    result['gallery_label'] = np.array([result['gallery_label']])
    result['gallery_path'] = np.array([result['gallery_path']])

    result['query_label'] = np.array([result['query_label']])
    result['query_path'] = np.array([result['query_path']])

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    multi = os.path.isfile('multi_query.mat')
    
    query_feature = query_feature.cuda(0)
    gallery_feature = gallery_feature.cuda(0)
    # import pdb; pdb.set_trace()
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    #print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    info = 'Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100)
    print('validation  ===> ',info)
    recall_1.append(CMC[0]*100)
    recall_5.append(CMC[4]*100)
    return  recall_1, recall_5
    # print(opt.name)
    # result = 'result.txt'
    # os.system('python evaluate_gpu.py | tee -a %s'%result)
# def main()
