#%%
from types import DynamicClassAttribute
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.utils as utils
import os
import torchvision
from torch.utils.data import DataLoader, Dataset
from collections import Iterable
from torchvision import transforms
import pandas
from collections import Iterable
from pathlib import Path
import random
import tqdm 

from py.dataprocessing.KTTIdataset import KTTIDataset
from py.visualization.box_ops import boxes_value_to_percentage

#%%
class KTTIFragmentDataset:
    """
    创建一个输出KITTI片段的数据集
    inputs:
        path: 标签所在文件夹路径
        img_path：图像所在文件夹路径
        partial: 取值
        exclusive_raws：去掉源标签文件中某行（整行），key为列名，value为要匹配的值，例如{type: ["Cyclist", "Misc"]}
        exclusive_cats：去掉源标签文件中某列(整列)，例如：["dimensions_0", "dimensions_1"]
        img_target_trans：对输出进行转化
    """
    def __init__(
        self, path, img_path, len, partial={}, exclusive_raws={}, exclusive_cats={}, im_target_trans=None
    ):
        self.ds = KTTIDataset(path, img_path, partial=partial, return_frame=True, exclusive_raws=exclusive_raws, 
                        exclusive_cats=exclusive_cats)
        self.len = len
        self.__im_tar_transformer = im_target_trans
        # self._registered_func_fragment = []
        masks = [np.ones(len, dtype=np.int) for len in self.ds.lens]
        if self.len > 1:  # 计算出可以使用的 index 避开不够长的
            for mask in masks:
                mask[-(self.len - 1):] = 0 
        masks = np.concatenate(masks, axis=0)    
        self.index = np.where(masks > 0)[0]  # 保存所有可以选择的index的数值(长度小于原有的)
            
    def type_id_to_str(self):
        return self.ds.type_id

    def __len__(self):
        return len(self.index) - 1

    # def register_imgs_hook(self, func):
    #     self._registered_func_fragment.append(func)

    def __getitem__(self, idx):
        idxs = self.index[idx]  # 获取index 避开长度不够的地方
        if isinstance(idxs, Iterable):
            raise RuntimeError("fragment dataset can't use slice operation.")
        else:
            idx = idxs
            imgs = []
            targets = []
            for i in range(self.len): 
                img, target = self.ds.__getitem__(idx + i)
                imgs.append(img)
                targets.append(target)
            # for func in self._registered_func_fragment:
            #     func(imgs, targets)
            if self.__im_tar_transformer is not None:
                imgs, targets = self.__im_tar_transformer(imgs, targets)
            return imgs, targets

class FragmentDL:
    def __init__(self, ds, shuffle=False) -> None:
        self.ds = ds
        self.idx = 0
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        if self.n < len(self):
            if self.shuffle:
                idx =  random.randint(0, len(self.ds))
                return self.ds[idx]
            else:
                res = self.idx
                self.idx += 1
                return self.ds[res]
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.ds)

#%%
if __name__ == '__main__':
    import time
    from tqdm import tqdm
    ds = KTTIFragmentDataset(
            path=Path(r"E:\KTTI\training\label_02"),
            img_path=Path(r"E:\KTTI\trackong_image\training\image_02"),
            len=3,
            exclusive_raws={'track_id':[-1], 'type':['Misc', 'Cyclist', 'Person', 'Tram', 'Truck', 'Van']}, 
            exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
        )
#%%
    dl = tqdm(FragmentDL(ds, shuffle=False))
    for imgs, targets in dl:
        dl.set_description("说一些无关紧要的话")
        # dl.set_description("不知道")
        time.sleep(0.1)