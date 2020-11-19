#%%
import pycocotools
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from collections import Iterable
import matplotlib.pyplot as plt
import os
#%%
class CocoDataset(Dataset):
    '''
    构建一个coco数据集(可选取类别)
    cats_name -> cat_ids -> img_ids -> ann_ids -> anns
                                    -> imgs
    ''' 
    instances = {}
    def __new__(cls, *args, **kwargs):
        if 'targets_path' in kwargs:
            path = kwargs['targets_path']
        else:
            path = args[0]
        if not os.path.exists(path):
            raise RuntimeError("Please fill the targets file path.")
        if path not in cls.instances:
            print('Creating coco dataset from anns fils: {}'.format(path))
            cls.instances[path] = {
                'instances': super().__new__(cls),
                'coco_obj': pycocotools.coco.COCO(path)
            }
        return cls.instances[path]['instances']

    # 初始化
    # image_path: 图片所在路径
    # image_transform: 图片变换类
    # target_transform: 标签变换类
    # cat_name: 可选取部分类别，默认为全选
    # super_cat: 可选区部分类别(父类)，默认为全选
    def __init__(self, targets_file, image_path, image_transform=None, target_transform=None, cat_name=[], super_name=[]):
        """
        inputs:
            targets_file: 标签所在路径
            image_path: 图片所在路径
            image_transform: 图片变换类
            target_transform: 标签变换类
            cat_name: 可选取部分类别，默认为全选
            super_cat: 可选区部分类别(父类)，默认为全选
        """
        self.coco = CocoDataset.instances[targets_file]['coco_obj']
        self.cat_ids = self.coco.getCatIds(catNms=cat_name, supNms=super_name)
        img_id_lis = [self.coco.getImgIds(catIds=[cid]) for cid in self.cat_ids]
        img_ids = np.array([x for l in img_id_lis for x in l])
        # 筛选掉iscrowd为1的图片id
        anns = self.coco.getAnnIds(iscrowd=True)
        crowd_img_ids = np.array(list(set([ann['image_id'] for ann in self.coco.loadAnns(anns)])))
        crowd_img_ids = [val for val in img_ids if val in crowd_img_ids]  # 取交集
        index = []
        for crowd_img_id in crowd_img_ids:
            index.append(np.where(img_ids == crowd_img_id)[0][0])
        self.img_ids = list(np.delete(img_ids, index))
        self.image_transform = image_transform
        self.target_transform = target_transform
id()        self.image_path = image_path

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img = Image.open(
            os.path.join(self.image_path, img_info['file_name'])
        ).convert("RGB")  # 转换为RGB格式

        if self.image_transform is not None:
            img = self.image_transform(img)  # 转换
        # 获取标签信息
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        # 筛选掉不在选择范围(cat_name)里面的anns
        after_filter_anns = []
        for ann in anns:
            if ann['category_id'] in self.cat_ids:
                after_filter_anns.append(ann)
        if self.target_transform is not None:
            after_filter_anns = self.target_transform(after_filter_anns)  # 转换
        return img, after_filter_anns       

    def __len__(self):
        return len(self.img_ids)
# %%
if __name__ == '__main__':
    anns_file = "/share/data/annotations/instances_train2017.json"
    image_path = "/share/data/train2017"
    ds = CocoDataset(anns_file, image_path)
    ds2 = CocoDataset(anns_file, image_path)
    print("Is id same? {}".format(id(ds) == id(ds2)))