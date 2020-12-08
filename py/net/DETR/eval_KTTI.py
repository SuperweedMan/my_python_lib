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
from engine import train_one_epoch

import models
from PIL import Image
import os
import json
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
# 数据集
dataset_train = KTTIdataset.build_KTTIDataset()

data_loader_train =DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)
#%%
checkpoint = torch.load('./KTTI_epoch_49', map_location= lambda storage, loc : storage.cuda())
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
if 'epoch' in checkpoint:
    CONFIG.start_epoch = checkpoint['epoch']
model.to(device)

#%%
model.eval()
criterion.eval()
for frame_num, (samples, targets) in enumerate(data_loader_train):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(samples)

    output['pred_logits'] = torch.nn.functional.softmax(output['pred_logits'], dim=2)[0,:,:-1]
    keep_idx = torch.max(output['pred_logits'], axis=-1).values > 0.7
    output['pred_logits'] = output['pred_logits'][keep_idx]
    output['pred_boxes'] = output['pred_boxes'][:, keep_idx]
    # 准备模型预测输出 imgs 
    output['pred_boxes'] = output['pred_boxes'].detach()
    img_filenames = [os.path.join(
                                        '{:0>4d}'.format(target['fragment'].cpu().numpy()), 
                                        '{:0>6d}'.format(target['frame'].cpu().numpy()) + '.png'
                                    ) for target in targets]
    path = '/share/data/KTTI/trackong_image/training/image_02'
    imgs_path = [os.path.join(path, filename) for filename in img_filenames]
    imgs = [torchvision.io.read_image(img_path) for img_path in imgs_path]

    
    for index, target in enumerate(targets):
        # fig = plt.figure(figsize=(12.24, 3.7))
        # ax = fig.add_subplot(1,1,1)
        # ax_operator = AxesOperations(ax)
        # ax_operator.labeled_boxs(output, 'cxcywh', KTTIdataset.KTTIDataset.type_id, target['size'],

        #     )
        output['size'] = target['size']
        del output['aux_outputs']
        output_json = {k: v.detach().cpu().numpy().tolist() for k, v in output.items()}
        path = '/share/data/KTTI_predictions/fragment_{:0>4d}-frame{:0>6d}.json'.format(target['fragment'], target['frame'])
        with open(path, 'w') as f:
            json.dump(output_json, f)


        # # print(target['orig_size'], target['size'], target['boxes'])
        # # targets
        # bbox = box_ops.boxes_percentage_to_value(target['boxes'], target['size'])
        # bbox = torchvision.ops.box_convert(bbox, 'cxcywh', "xywh")
        # # for box in bbox:
        # #     rect = plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor = 'red',linewidth=1)
        # #     ax.add_patch(rect)
        # # outputs
        # boxes = output['pred_boxes'][index]
        # boxes = box_ops.boxes_percentage_to_value(boxes, target['size'])
        # boxes = torchvision.ops.box_convert(boxes, 'cxcywh', "xywh")
        # for i, box in enumerate(boxes):
        #     rect = plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor = 'green',linewidth=1)
        #     ax.add_patch(rect)
        #     ax.text(rect.xy[0], rect.xy[1], "{0:.3f}".format(float(output['pred_logits'][i].max().detach().cpu().numpy())),
        #             va='top', ha='left', fontsize=6, color='black',
        #             bbox=dict(facecolor='g', joinstyle='round', alpha=0.6, lw=0, pad=0))
        # 显示图像
        # plt.imshow(
        #     torch.nn.functional.interpolate(
        #         imgs[index][None], size=samples.tensors[index].shape[-2:]
        #         ).squeeze(0).permute(1, 2, 0).numpy()
        # )
        # plt.imshow(imgs[index][None].squeeze(0).permute(1, 2, 0).numpy())
        # plt.savefig('/share/data/KTTI_predictions/fragment_{:0>4d}-frame{:0>6d}.jpg'.format(target['fragment'], target['frame']))