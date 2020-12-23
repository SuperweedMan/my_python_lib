#%%
import os
os.environ['DISPLAY']=":1"
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
import torch.nn.functional as F
import torchvision
import easydict
from torch import nn
from datasets.coco import build as build_coco
from models.util import misc as utils
from config import CONFIG
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from engine import train_one_epoch

import models
from PIL import Image
import json
import matplotlib.pyplot as plt
from py.visualization import box_ops
from py.dataprocessing import KTTIdataset
from py.visualization.plt_operations import AxesOperations
from py.dataprocessing.KTTIdataset import CropAnns, ToTensor, ToPercentage, CovertBoxes,Compose
from py.net.DETR.models.util.misc import nested_tensor_from_tensor_list
from py.visualization.plt_operations import AxesOperations

#%%
device = torch.device(CONFIG.device)
# 创建模型
model, criterion, postprocessors = models.build_model(CONFIG)

checkpoint = torch.load('./KTTI_epoch_5', map_location= lambda storage, loc : storage.cuda())
model.load_state_dict(checkpoint['model'])
model.to(device)

# 图像预处理
img_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),       
])
target_transformer = transforms.Compose([
    CropAnns(['boxes', 'labels', 'size', 'orig_size', 'fragment', 'frame', 'track_id']),
    ToTensor()
])
im_target_trans = Compose([
    ToPercentage(),
    CovertBoxes('xyxy', 'cxcywh')
])

def img_preprocessing(module, inputs):
    if not module.training:
        inputs = list(inputs)
        for idx in range(len(inputs)):
            inputs[idx] = img_transformer(inputs[idx])
        inputs = nested_tensor_from_tensor_list(inputs)
        return inputs.to(device)
    else:
        raise RuntimeError("img_preprocess just for eval model.")

img_preprocess_handle = model.register_forward_pre_hook(img_preprocessing)
#%%
path = "/share/data/KTTI/trackong_image/testing/image_02/0006"
imgs_name = os.listdir(path)
imgs_name.sort()
model.eval()
# 画图
fig = plt.figure(figsize=(12.42, 3.75))
ax = fig.add_axes([0, 0, 1, 1])
ax_operator = AxesOperations(ax)
plt.ion()
for img_name in imgs_name:
    img = Image.open(os.path.join(path, img_name))
    output = model(img)
    ax.imshow(img)
    # 处理模型输出
    output['pred_logits'] = torch.nn.functional.softmax(output['pred_logits'], dim=2)[0,:,:-1]
    keep_idx = torch.max(output['pred_logits'], axis=-1).values > 0.7
    output['pred_logits'] = output['pred_logits'][keep_idx]
    output['pred_boxes'] = output['pred_boxes'][:, keep_idx]
    # 准备模型预测输出 imgs 
    output['pred_boxes'] = output['pred_boxes'].detach().cpu()
    img_size = list(img.size)
    img_size.reverse()
    ax_operator.labeled_boxs(
        output,
        'cxcywh', 
        ['Car', 'Pedestrain'], 
        torch.tensor(img_size),
    )
    plt.pause(0.1)
    plt.cla()
plt.ioff()