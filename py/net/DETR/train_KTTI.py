#%%
import torch
import torch.nn.functional as F
import torchvision
import easydict
from torch import nn
from datasets.coco import build as build_coco
from models.util import misc as utils
from config import CONFIG
from torch.utils.data import DataLoader, DistributedSampler
from engine import train_one_epoch,eval_one_epoch

import models
from PIL import Image
import os
import matplotlib.pyplot as plt
from py.visualization import box_ops
from py.dataprocessing import KTTIdataset
#%%
device = torch.device(CONFIG.device)
# 创建模型
model, criterion, postprocessors = models.build_model(CONFIG)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)
# 收集权重
param_dicts = [ 
        # backbone 以外的权重 
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        # backbine 的权重
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": CONFIG.lr_backbone,
        },
    ]
# 优化器
optimizer = torch.optim.AdamW(param_dicts, lr=CONFIG.lr,
                                  weight_decay=CONFIG.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG.lr_drop)
# 数据集
dataset_train = KTTIdataset.build_KTTIDataset()
data_loader_train =DataLoader(dataset_train, batch_size=10, shuffle=True, collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)
#%%
# 载入
# checkpoint = torch.load(CONFIG.resume)
# # del checkpoint['model']['class_embed.weight']
# # del checkpoint['model']['class_embed.bias']
# model.class_embed = torch.nn.Linear(
#     in_features=model.class_embed.in_features,
#     out_features=92,  # ['Car', 'Pedestrian'] + background
#     bias=True
# )
# model.load_state_dict(checkpoint['model'])
# model.class_embed = torch.nn.Linear(
#     in_features=model.class_embed.in_features,
#     out_features=3,  # ['Car', 'Pedestrian'] + background
#     bias=True
# )
checkpoint = torch.load('./KTTI_epoch_49', map_location= lambda storage, loc : storage.cuda())
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
if 'epoch' in checkpoint:
    CONFIG.start_epoch = checkpoint['epoch']
model.to(device)

#%%
FIFO = []
for epoch in range(CONFIG.start_epoch, CONFIG.epochs):
    # train_stats = train_one_epoch(
    #         model, criterion, data_loader_train, optimizer, device, epoch,
    #         CONFIG.clip_max_norm)
    # lr_scheduler.step()

    eval_stats = eval_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, CONFIG.clip_max_norm
        )

    # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                  'epoch': epoch}

    log_stats = {**{f'train_{k}': v for k, v in eval_stats.items()}, 'epoch': epoch}
    FIFO.append(log_stats)
#%%
torch.save(
    {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    },
    './KTTI_epoch_49'
)