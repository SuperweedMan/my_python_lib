#%%
import torch
import torch.nn.functional as F
import torchvision
import easydict
from torch import nn
# from datasets.coco import build as build_coco
from models.util import misc as utils
from config import CONFIG
from torch.utils.data import DataLoader, DistributedSampler
from engine import train_one_epoch,eval_one_epoch

import models
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
from py.visualization import box_ops
from py.dataprocessing import KTTIdataset
from py.visualization.plt_operations import AxesOperations
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
#%%
# 数据集
split_position = 0.7
dataset_train = KTTIdataset.build_KTTIDataset(partial={'front': split_position}, label_path=Path(r"E:\KTTI\training\label_02"),
            img_path=Path(r"E:\KTTI\trackong_image\training\image_02"))
data_loader_train =DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)

dataset_eval = KTTIdataset.build_KTTIDataset(partial={'behind': split_position}, label_path=Path(r"E:\KTTI\training\label_02"),
            img_path=Path(r"E:\KTTI\trackong_image\training\image_02"))
data_loader_eval =DataLoader(dataset_eval, batch_size=8, shuffle=True, collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)

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
checkpoint = torch.load(Path(r"E:\kitti_parameter\KTTI_epoch_19"), map_location= lambda storage, loc : storage.cuda())
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
if 'epoch' in checkpoint:
    CONFIG.start_epoch = checkpoint['epoch']
model.to(device)
FIFO = checkpoint['FIFO']
#%%
# FIFO = []
for epoch in range(CONFIG.start_epoch, CONFIG.epochs):
    train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            CONFIG.clip_max_norm)
    lr_scheduler.step()

    eval_stats = eval_one_epoch(
            model, criterion, data_loader_eval, optimizer, device, epoch, CONFIG.clip_max_norm
        )

    log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
    log_stats_eval = {**{f'train_{k}': v for k, v in eval_stats.items()}, 'epoch': epoch}
    FIFO.append((log_stats_train, log_stats_eval))
#%%
# checkpoint = torch.load('./KITTI_epoch_13')
# FIFO = checkpoint['FIFO']
train_loss, eval_loss = zip(*FIFO) 
train_loss = [loss['train_loss'] for loss in train_loss]
eval_loss = [loss['train_loss'] for loss in eval_loss]
#%%
fig, ax = plt.subplots(1)
ax_operator = AxesOperations(ax)
ax_operator.plot(train_loss, smooth_arg=0, c='darkorange')
ax_operator.plot(eval_loss, smooth_arg=0, c='blue')
fig.show()
#%%
torch.save(
    {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'FIFO': FIFO,
    },
    './KITTI_epoch_13'
)
#%%
def callbackfunc(module, input):
    print(type(module))

handle = model.register_forward_pre_hook(callbackfunc)