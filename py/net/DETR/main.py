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

import models
from PIL import Image
import os
import matplotlib.pyplot as plt
from py.visualization import box_ops
#%%
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


dataset_train = build_dataset(image_set='train', args=CONFIG)
# sampler_train = torch.utils.data.RandomSampler(dataset_train)
# batch_sampler_train = torch.utils.data.BatchSampler(
#     sampler_train, CONFIG.batch_size, drop_last=True)
# data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
#                                 collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)
#%%
data_loader_train =DataLoader(dataset_train, shuffle=False, collate_fn=utils.collate_fn, num_workers=CONFIG.num_workers)
#%%
model, criterion, postprocessors = models.build_model(CONFIG)
device = torch.device(CONFIG.device)
checkpoint = torch.load(CONFIG.resume)

model.load_state_dict(checkpoint['model'])
model.to(device)
#%%
atten = {}
feature_shape = None
def encoder_atten(module, input, output):
    global atten
    # print(module)
    atten['encoder'] = [x.detach().cpu() for x in output]  # atten_output, atten_weight
    # print(atten)            

def decoder_atten(module, input, output):
    global atten
    # print(module)
    atten['decoder'] = [x.detach().cpu() for x in output]  # atten_output, atten_weight

def print_shape(module, input, output):
    global feature_shape
    feature_shape = output.shape
    print(output.shape)

# model.transformer.encoder.layers[5].self_attn.register_forward_hook(print_atten)
model.transformer.encoder.layers[5].self_attn.register_forward_hook(encoder_atten)
model.transformer.decoder.layers[5].multihead_attn.register_forward_hook(decoder_atten)
model.backbone[0].body.layer4[2].conv3.register_forward_hook(print_shape)
#%%
for index, (samples, targets) in enumerate(data_loader_train):
        if index < 1:
            continue
        samples = samples.to(device)
        output = model(samples)
        # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # # keep_idx = probas.max(-1).values > 0.7
        output['pred_logits'] = torch.nn.functional.softmax(output['pred_logits'], dim=2)[0,:,:-1]

        keep_idx = torch.max(output['pred_logits'], axis=-1).values > 0.7
        output['pred_logits'] = output['pred_logits'][keep_idx]
        output['pred_boxes'] = output['pred_boxes'][:, keep_idx, :]


        # show boxes
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for target in targets:
            # bbox = torchvision.ops.box_convert(target['boxes'], 'cxcywh', 'xyxy')
            print(target['orig_size'], target['size'], target['boxes'])
            bbox = box_ops.boxes_percentage_to_value(target['boxes'], target['size'])
            bbox = torchvision.ops.box_convert(bbox, 'cxcywh', "xywh")
            for box in bbox:
                rect = plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor = 'red',linewidth=1)
                ax.add_patch(rect)
            # ax.text(rect.xy[0]+10, rect.xy[1]+10, "score: {0:1.2f}".format(score),
            #             va='top', ha='left', fontsize=6, color='black',
            #             bbox=dict(facecolor='r', joinstyle='round', alpha=0.35, lw=0))
        for boxes in output['pred_boxes'].detach().cpu():
            boxes = box_ops.boxes_percentage_to_value(boxes, target['size'])
            boxes = torchvision.ops.box_convert(boxes, 'cxcywh', "xywh")
            for box in boxes:
                rect = plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor = 'green',linewidth=1)
                ax.add_patch(rect)
        # show imgs
        img_filenames = ['{0:012d}.jpg'.format(int(target['image_id'].cpu().numpy())) for target in targets]
        path = '/share/data/train2017'
        imgs_path = [os.path.join(path, filename) for filename in img_filenames]
        imgs = [torchvision.io.read_image(img_path) for img_path in imgs_path]
        # big_img = torchvision.utils.make_grid(imgs)
        # plt.imshow(samples.tensors[0].cpu().permute(1, 2, 0).numpy())
        print(samples.tensors[0].shape)
        for index, img in enumerate(imgs):
            plt.imshow(
                torch.nn.functional.interpolate(
                    img[None], size=samples.tensors[index].shape[-2:]
                    ).squeeze(0).permute(1, 2, 0).numpy()
            )
        plt.show()
        
        # show atten
        def my_normalize(a, b):
            return 1 / b * a
        for k, v in atten.items():
            v[1][0].map_(v[1].max(), my_normalize)
        # atten[1][0].map_(atten[1].max(), my_normalize)
        # atten_maps = atten[1].resize(1, 100, *feature_shape[-2:]).permute(1, 0, 2, 3)
            atten_maps = v[1].resize(1, v[1].shape[-2], *feature_shape[-2:]).permute(1, 0, 2, 3)
            if k == 'decoder':
                f = atten_maps[torch.where(keep_idx == True)]  # (boxes(2), channel(1), h, w)
                plt.imshow(
                    torchvision.utils.make_grid(f, pad_value=1).permute(1, 2, 0).numpy(),
                    cmap=plt.cm.hot, interpolation='none'
                )
                plt.show()
            atten_maps = torchvision.utils.make_grid(atten_maps, nrow=int(feature_shape[-1]), pad_value=1)
            plt.imshow(atten_maps.permute(1, 2, 0).numpy(), cmap=plt.cm.hot, interpolation='none')
            plt.show()

        v=atten['encoder']
        atten_maps = v[1].resize(1, v[1].shape[-2], *feature_shape[-2:]).permute(1, 0, 2, 3)
        box = []
        for focus in f:
            focus = focus[0].reshape(-1)
            atten_map = atten_maps.clone().detach()
            for index, weight in enumerate(focus):
                atten_map[index].mul_(weight)
            box.append(atten_map.sum(0))
        boxes_img = torchvision.utils.make_grid(box, pad_value=1)
        plt.imshow(boxes_img.permute(1, 2, 0).numpy(), cmap=plt.cm.hot, interpolation='none')
        break


# %%
