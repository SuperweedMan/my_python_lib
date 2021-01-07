#%%
import enum
from types import MappingProxyType
import torch
import torchvision
from torchvision import transforms
import copy
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib
from collections import Iterable
from py.visualization.box_ops import boxes_value_to_percentage
from py.visualization.plt_operations import AxesOperations
import random
import cv2
import io
from PIL import Image
from typing import List

#%%
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, anns):
        for t in self.transforms:
            images, anns = t(images, anns)
        return images, anns

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Shelter(object):
    def __init__(self, box_from, size=(0.1, 0.5), proportion=0.5):
        assert box_from in ('xyxy', 'cxcywh', 'xywh')
        self.orig_form = box_from
        self.size = size
        self.proportion = proportion
        self.toPIL = transforms.Compose([transforms.ToPILImage()])
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __call__(self, imgs, anns):
        assert isinstance(imgs, list) and isinstance(imgs[0], torch.Tensor)
        assert imgs[0].ndim == 3
        img_size =  tuple(( i / 100. for i in tuple(imgs[0].shape[-2:])))
        idx = random.randint(1, len(imgs) - 1)  # 不抽取第一张
        fig, ax = plt.subplots()
        ax.imshow(self.toPIL(imgs[idx]), aspect="equal", interpolation='none')
        proportion = round(self.proportion * len(anns[idx]['boxes']))
        size = random.uniform(*self.size)
        labels = random.sample(list(anns[idx]['boxes']), proportion)
        labels = [torchvision.ops.box_convert(
                                    boxes, 
                                    self.orig_form, 
                                    'cxcywh').cpu().numpy() for boxes in labels]
        for boxes in labels:
            circle = plt.Circle((boxes[0],boxes[1]), np.min(boxes[-2:]) * size / 2., color = 'black')
            ax.add_patch(circle)
        plt.axis("off")
        # 去除图像周围的白边
        channels, height, width = imgs[idx].shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format='png', dpi=100)
        buffer_.seek(0)
        imgs[idx] = Image.open(buffer_).convert("RGB")
        imgs[idx] = self.toTensor(imgs[idx])
        buffer_.close()
        plt.close()
        # fig.canvas.draw()
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # imgs[idx] = torch.tensor(data).permute(2, 1, 0)
        return imgs, anns


class MotionPlur(object):
    def __init__(self, degree=(20, 50), angle=(20, 100)):
        self.degree = degree
        self.angle =angle

    def __call__(self, imgs, anns):
        assert isinstance(imgs, list) and isinstance(imgs[0], torch.Tensor)
        assert imgs[0].ndim == 3
        # 抽取一帧
        degree = random.randint(*self.degree)
        angle = random.randint(*self.angle)
        idx = random.randint(1, len(imgs) - 1)  # 不抽取第一张
        img = imgs[idx].cpu().permute(1, 2, 0).numpy()
            # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高 
        M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1) 
        motion_blur_kernel = np.diag(np.ones(degree)) 
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree)) 
        motion_blur_kernel = motion_blur_kernel / degree 
        blurred = cv2.filter2D(img, -1, motion_blur_kernel) 
        # convert to uint8 
        cv2.normalize(blurred, blurred, 0, 1, cv2.NORM_MINMAX) 
        # blurred = np.array(blurred, dtype=np.uint8)
        imgs[idx] = torch.tensor(blurred.transpose(2, 0, 1))
        return imgs, anns

class Disturbance(object):
    def __init__(self, disturbances:List=[]) -> None:
        self.pool = disturbances

    def __call__(self, imgs, anns):
        if len(self.pool) == 0:
            return imgs, anns
        else:
            dicturbance = random.sample(self.pool, 1)[0]
            imgs, anns = dicturbance(imgs, anns)
            return imgs, anns

class ToPercentage(object):
    def __init__(self):
        pass

    def __call__(self, imgs, anns):
        def recursion_img(imgs):  # 迭代读取直到图片层，读取图片大小
            if isinstance(imgs, Iterable) and (not isinstance(imgs, torch.Tensor)) :
                return recursion_img(imgs[0])
            else:
                return imgs.shape[-2:]
        h, w = recursion_img(imgs)

        def recursion_ann(anns):
            for item in anns:
                if isinstance(item, dict):
                    for name in item:
                        if name == 'boxes':
                            item[name] = boxes_value_to_percentage(item[name], (h, w))
                else:
                    recursion_ann(item)

        recursion_ann(anns)
        return imgs, anns

class ToTensor(object):
    """Change anns to Tensor"""
    def __init__(self) -> None:
        self.img_totensor = transforms.ToTensor()

    def __call__(self, imgs, anns): 
        def recursion_img(imgs):  # 迭代读取直到图片层，转化图片为tensor
            if isinstance(imgs, Iterable):
                for idx, item in enumerate(imgs):
                    imgs[idx] = recursion_img(item)
            else:
                return self.img_totensor(imgs)

        def recursion_ann(anns):
            for item in anns:
                if isinstance(item, dict):
                    for name in item:
                        item[name] = torch.tensor(item[name])
                else:
                    recursion_ann(item)
        if isinstance(imgs, Iterable) and type(imgs) != torch.Tensor:
            recursion_img(imgs)
        else:
            imgs = self.img_totensor(imgs)
        recursion_ann(anns)
        return imgs, anns

class CropAnns(object):
    """Crop anns"""
    def __init__(self, names) -> None:
        self.names = names
    
    def __call__(self, imgs, anns):
        def recursion(anns):
            for item in anns:
                if isinstance(item, dict):  # 直接修改
                    for name in list(item.keys()):
                        if name in self.names:
                            del item[name]
                else:
                    recursion(item)

        recursion(anns)
        return imgs, anns

class CovertBoxes(object):
    """Covert anns' form from orig_form to target_form"""
    def __init__(self, orig_form, target_form):
        assert orig_form in ('xyxy', 'cxcywh', 'xywh')
        assert target_form in ('xyxy', 'cxcywh', 'xywh')
        self.orig_form = orig_form
        self.target_form = target_form

    def __call__(self, imgs, anns):
        """inputs: 
            list[img] or img
            list[list[dict{...}]] or list[dict{...}]
        """
        def recursion(anns):
            for item in anns:
                if isinstance(item, dict):  # 直接修改
                    for name in item:
                        if name == 'boxes':
                            item[name] = torchvision.ops.box_convert(
                                    item[name], 
                                    self.orig_form, 
                                    self.target_form
                                )
                else:
                    recursion(item)
        recursion(anns)
        return imgs, anns

#%%
if __name__ == "__main__":
    import KITTI_fragment_ds
    from pathlib import Path
    ds = KITTI_fragment_ds.KTTIFragmentDataset(
            # partial={'front': 0.1},
            path=Path(r"E:\KTTI\training\label_02"),
            img_path=Path(r"E:\KTTI\trackong_image\training\image_02"),
            len=3,
            exclusive_raws={'track_id':[-1], 'type':['Misc', 'Cyclist', 'Person', 'Tram', 'Truck', 'Van']}, 
            exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
            im_target_trans=Compose([
                CropAnns(['fragment', 'type']),
                ToTensor(),
                Disturbance([
                    MotionPlur(degree=(10, 30)),
                    Shelter('xyxy', proportion=1.0, size=(0.5, 0.7))
                ]),
                ToPercentage(),
                CovertBoxes('xyxy', 'cxcywh'),
            ])
        )
#%%
if __name__ == "__main__":
    from torchvision.utils import make_grid
    from torchvision import transforms
    toPIL = transforms.Compose([transforms.ToPILImage()])
    toTensor = transforms.Compose([transforms.ToTensor()])
    dl = KITTI_fragment_ds.FragmentDL(ds, shuffle=False)
    for imgs, targets in dl:
        plt.ion()
        img = make_grid(imgs, nrow=1, padding=0)
        channels, height, width = img.shape
        img = toPIL(img)
        
        fig, ax = plt.subplots()
        ax.imshow(img, aspect="equal", interpolation='none')
        # circ = plt.Circle((0,0), 100, color = 'black')
        # ax.add_patch(circ)
        plt.axis("off")
        # 去除图像周围的白边
        # height, width, channels = im.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        # plt.savefig('./test.png')
        plt.show()