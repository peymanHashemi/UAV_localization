from torchvision import transforms
from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .autoaugment import ImageNetPolicy
import torch
from .queryDataset import RotateAndCrop, RandomCrop, RandomErasing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

def make_dataset(opt):
    transform_train_list = []
    transform_satellite_list = []
    seg_transform_train_list = []
    seg_transform_satellite_list = []
    if "uav" in opt.rr:
        transform_train_list.append(RotateAndCrop(0.5))
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(0.5))
    transform_train_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    seg_transform_train_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_satellite_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]
    seg_transform_satellite_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    if "uav" in opt.ra:
        transform_train_list = transform_train_list + \
            [transforms.RandomAffine(180)]
        
        seg_transform_train_list = seg_transform_train_list + \
            [transforms.RandomAffine(180)]
        
    if "satellite" in opt.ra:
        transform_satellite_list = transform_satellite_list + \
            [transforms.RandomAffine(180)]
        seg_transform_satellite_list = seg_transform_satellite_list + \
            [transforms.RandomAffine(180)]
    if "uav" in opt.re:
        transform_train_list = transform_train_list + \
            [RandomErasing(probability=opt.erasing_p)]
        seg_transform_train_list = seg_transform_train_list + \
            [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_satellite_list = transform_satellite_list + \
            [RandomErasing(probability=opt.erasing_p)]
        seg_transform_satellite_list = seg_transform_satellite_list + \
            [RandomErasing(probability=opt.erasing_p)]
    if "uav" in opt.cj:
        transform_train_list = transform_train_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
        seg_transform_train_list = seg_transform_train_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
    if "satellite" in opt.cj:
        transform_satellite_list = transform_satellite_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
        seg_transform_satellite_list = seg_transform_satellite_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list
        seg_transform_train_list = [ImageNetPolicy()] + seg_transform_train_list
    last_aug = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list += last_aug
    transform_satellite_list += last_aug

    seg_transform_train_list += last_aug
    seg_transform_satellite_list += last_aug

    transform_val_list += last_aug

    print(transform_train_list)
    print(transform_satellite_list)

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list),
        'seg_train': transforms.Compose(seg_transform_train_list),
        'seg_satellite': transforms.Compose(seg_transform_satellite_list)
        }

    # # custom Dataset
    # if opt.segmentaion:


    image_datasets = Dataloader_University(
        opt.data_dir, transforms=data_transforms, seg_root=opt.seg_dir)
    samper = Sampler_University(
        image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                            sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    dataset_sizes = {x: len(image_datasets) *
                    opt.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    # print('===============================',len(image_datasets))
    # a = (next(iter(dataloaders))[3][0][0]).permute(1,2,0).numpy()
    # a = image_datasets[0]
    # a -= a.min()
    # a /= a.max()
    # a *= 255
    # a = a.astype(np.uint8)
    # # print('aaaaaaaaaaaaaaaa',a)
    # cv2.imwrite('./test1.png', a)
    # plt.imshow((next(iter(dataloaders))[0][0][0]).permute(1,2,0))
    # plt.show()
    # import pdb; pdb.set_trace()
    return dataloaders, class_names, dataset_sizes

    

    #     data_transforms = {
    #     'train': transforms.Compose(transform_train_list),
    #     'val': transforms.Compose(transform_val_list),
    #     'satellite': transforms.Compose(transform_satellite_list)}

    #     image_datasets = Dataloader_University(
    #         opt.data_dir, transforms=data_transforms)
    #     samper = Sampler_University(
    #         image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    #     dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
    #                                             sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    #     dataset_sizes = {x: len(image_datasets) *
    #                     opt.sample_num for x in ['satellite', 'drone']}
    #     class_names = image_datasets.cls_names
    
    #     seg_transforms = {
    #     'train': transforms.Compose(transform_train_list),
    #     'val': transforms.Compose(transform_val_list),
    #     'satellite': transforms.Compose(transform_satellite_list)}

    #     seg_datasets = Dataloader_University(
    #         opt.seg_dir, transforms=seg_transforms)
    #     seg_samper = Sampler_University(
    #         seg_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    #     seg_dataloaders = torch.utils.data.DataLoader(seg_datasets, batch_size=opt.batchsize,
    #                                             sampler=seg_samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    #     seg_dataset_sizes = {x: len(seg_datasets) *
    #                     opt.sample_num for x in ['satellite', 'drone']}
    #     class_names = seg_datasets.cls_names
    #     seg_class_names = seg_datasets.cls_names
    #     return dataloaders, class_names, dataset_sizes, seg_dataloaders, seg_dataset_sizes, seg_class_names
    # else: