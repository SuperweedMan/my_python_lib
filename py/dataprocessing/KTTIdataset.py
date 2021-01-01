#%%
# -----------------------------------------------------------------
# classes:
#    KTTIdataset: 根据输入参数创建一个KTTI的dateset实例。
#    CropAnns: 删除不需要的categories。
# functions:
#    collate_to_list: dataloader组合函数。
# -----------------------------------------------------------------
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

from py.visualization.box_ops import boxes_value_to_percentage

#%%
def get_targets_from_files(targets_path, partial={}, exclusive_raws={}, exclusive_cats=[]):
    if not os.path.exists(targets_path):
        raise ValueError('wrong path')
    all_targets = {}
    label_names = os.listdir(targets_path)
    # 切片
    if len(partial) != 0:
        key, item = list(partial.items())[0]
        if key in ("front", "behind"):
            split_position = int(item * len(label_names))
            if key == 'front': 
                label_names = label_names[:split_position]
            else:
                label_names = label_names[split_position:]
        else:
            raise RuntimeError("error key name {}".format(key))
    for label_name in label_names: 
        # 读取txt文件并添加标签头
        column_name = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_0', 'bbox_1', 'bbox_2', 'bbox_3', 'dimensions_0', 'dimensions_1', 'dimensions_2', 'location_0', 'location_1', 'location_2', 'ration_y']
        labels = pandas.read_csv(os.path.join(targets_path, label_name), sep=' ', names=column_name)
        if len(exclusive_cats) != 0:
            labels = labels.drop(columns=exclusive_cats)  # 删除不需要的类别
        if len(exclusive_raws) != 0:
            for cat_name in exclusive_raws:
                labels = labels[(~labels[cat_name].isin(exclusive_raws[cat_name]))]  # 删除不需要的行
        all_targets[label_name[:4]] = labels
    return all_targets

class KTTIDataset(Dataset):
    """
    创建一个KTTI dataset
    """
    type_id =  ['Car', 'Cyclist', 'Misc', 'Pedestrian', 'Person', 'Tram', 'Truck', 'Van', 'DontCare'] 
    def __init__(self, path, img_path, partial={}, return_frame=True,  
                exclusive_raws={}, exclusive_cats=[], img_transformer=None, 
                target_transformer=None, im_target_trans=None):
        """
        input:
            path: <string> path of label files(*.txt).
            img_path: <string> path of images.
            return_frame: <bool> is a single frame image returned or a entire sequence of images.
            exclusive_raws: <dict{list[value]}>  a dict that contains lists which contains unnecessary raw.
            exclusive_cats: <list[string]> a dict that contains unnecessary categories.
        """
        assert isinstance(exclusive_raws, dict) and isinstance(exclusive_cats, list)
        exclusive_raws = { k:v if isinstance(v, list) else [v] for k, v in exclusive_raws.items()}
        all_targets = get_targets_from_files(path, partial, exclusive_raws=exclusive_raws, exclusive_cats=exclusive_cats)  # 返回的是pandas格式
        self.all_targets_dict = {}
        self.return_frame = return_frame
        self.imgs_path = img_path
        self._registered_func = []
        self.img_transformer = img_transformer
        self.target_transformer = target_transformer
        self.im_target_trans = im_target_trans
        if 'type' in exclusive_raws:
            for item in exclusive_raws['type']:
                if item in KTTIDataset.type_id:
                    KTTIDataset.type_id.remove(item)
        for fragment in all_targets:
            targets = []
            for frame in sorted(set(all_targets[fragment]['frame'])):
                cut_labels = all_targets[fragment][
                            all_targets[fragment]['frame'].isin([frame])]
                label_dict = {k:list(v) for k, v in dict(cut_labels).items()}
                label_dict['boxes'] = np.array([label_dict['bbox_0'], label_dict['bbox_1'], label_dict['bbox_2'], label_dict['bbox_3']]).T.tolist()
                label_dict.pop('bbox_0')
                label_dict.pop('bbox_1')
                label_dict.pop('bbox_2')
                label_dict.pop('bbox_3')
                label_dict['frame'] = frame
                label_dict['fragment'] = fragment
                label_dict['labels'] = [KTTIDataset.type_id.index(t) for t in label_dict['type']]
                targets.append(label_dict)
            self.all_targets_dict[fragment] = targets
        self.num = [len(v) for k, v in self.all_targets_dict.items()]
        if return_frame is True:
            self.all_targets_dict = [v for k, v in self.all_targets_dict.items()]
            self.lens = [len(v) for v in self.all_targets_dict]
            self.all_targets_dict = [x for l in self.all_targets_dict for x in l]

    def register_img_hook(self, func):
        self._registered_func.append(func)

    def __getitem__(self, idx):
        """
        input: 
            idx: <int> index 
        return:
            tuple (img, target)
            img: <undefine> image data processed by img_transformer.
                    default type <PIL.PngImage>
            target: <underfine> target processed by target_transformer.
                    default type <dict{'categoies name': value or list[values of targets]}>
        """
        if self.return_frame:
            targets = self.all_targets_dict[idx]
            if isinstance(targets, list):
                returns = []
                for target in targets:
                    img_path = os.path.join(self.imgs_path, target['fragment'], '{:0>6d}'.format(target['frame']) + '.png')
                    img = Image.open(img_path)
                    if len(self._registered_func) > 0:
                        for func in self._registered_func:
                            func(img)
                    w, h =img.size
                    target['size'] = (h, w)
                    target['orig_size'] = (h, w)
                    if self.target_transformer is not None:
                        target = self.target_transformer(target)
                    if self.img_transformer is not None:
                        img = self.img_transformer(img)
                    if self.im_target_trans is not None:
                        img, target = self.im_target_trans(img, target)
                    returns.append((img, target))
                return returns
            else:
                target = targets
                img_path = os.path.join(self.imgs_path, target['fragment'], '{:0>6d}'.format(target['frame']) + '.png')
                img = Image.open(img_path)
                if len(self._registered_func) > 0:
                    for func in self._registered_func:
                        func(img)
                w, h =img.size
                target['size'] = (h, w)
                target['orig_size'] = (h, w)
                if self.target_transformer is not None:
                    target = self.target_transformer(target)
                if self.img_transformer is not None:
                    img = self.img_transformer(img)
                if self.im_target_trans is not None:
                    img, target = self.im_target_trans(img, target)
                return img, target
        else:
            raise ValueError('The function to return a entire sequence is not yet implemented.')

    def __len__(self):
        return sum(self.num)

class KTTIFragmentDataset(KTTIDataset):
    def __init__(self, path, img_path, len, partial={}, exclusive_raws={}, exclusive_cats={}, img_transformer=None, target_transformer=None, im_target_trans=None):
        self.len = len
        super().__init__(path, img_path, partial={}, return_frame=True, exclusive_raws=exclusive_raws, 
                        exclusive_cats=exclusive_cats, img_transformer=img_transformer, 
                        target_transformer=target_transformer, im_target_trans=im_target_trans)

        masks = [np.ones(len, dtype=np.int) for len in self.lens]
        for mask in masks:
            mask[-(self.len - 1):] = 0 
        masks = np.concatenate(masks, axis=0)    
        self.index = np.where(masks > 0)[0]  # 保存所有可以选择的index的数值(长度小于原有的)
        self._registered_func_fragment = []

    def __len__(self):
        return len(self.index)

    def register_imgs_hook(self, func):
        self._registered_func_fragment.append(func)

    def __getitem__(self, idx):
        idxs = self.index[idx]  # 获取index 避开长度不够的地方
        if isinstance(idxs, Iterable):
            raise RuntimeError("fragment dataset can't use slice operation.")
        else:
            idx = idxs
            imgs = []
            targets = []
            for i in range(self.len): 
                img, target = super().__getitem__(idx)
                imgs.append(img)
                targets.append(target)
            for func in self._registered_func_fragment:
                func(imgs, targets)
            return imgs, targets



def collate_to_list(batch):
    return list(zip(*batch))



class CropAnns:
    def __init__(self, name=[]):
        self.name = name
    
    def __call__(self, anns):
        target = {}
        for name in anns:
            if name in self.name:
                if name == 'fragment':
                    target[name] = int(anns[name])
                else:
                    target[name] = anns[name]
        return target

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, anns):
        target = {}
        for name in anns:
            target[name] = torch.tensor(anns[name])
        return target

class ToPercentage:
    def __init__(self):
        pass

    def __call__(self, imgs, anns):
        h, w = imgs.shape[-2:]
        target = {}
        for name in anns:
            if name == 'boxes':
                target[name] = boxes_value_to_percentage(anns[name], (h, w))
            else:
                target[name] = anns[name]
        return imgs, target

class CovertBoxes:
    def __init__(self, orig_form, target_form):
        self.orig_form = orig_form
        self.target_form = target_form

    def __call__(self, imgs, anns):
        target = {}
        for name in anns:
            if name == 'boxes':
                target[name] = torchvision.ops.box_convert(
                        anns[name], 
                        self.orig_form, 
                        self.target_form
                    )
            else:
                target[name] = anns[name]
        return imgs, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def build_KTTIDataset(partial, label_path, img_path):
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
    
    ds = KTTIDataset(
        path= label_path,
        partial=partial,
        img_path=img_path, 
        exclusive_raws={'track_id':[-1], 'type':['Misc', 'Cyclist', 'Person', 'Tram', 'Truck', 'Van']}, 
        exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
        img_transformer=img_transformer,
        target_transformer=target_transformer,
        im_target_trans=im_target_trans
        )
    return ds

def build_eval_KTTIDataset(just_show=False):
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
    if just_show:
        ds = KTTIDataset(
            path='/share/data/KTTI/training/label_02', 
            img_path='E:\KTTI\trackong_image\training\image_02', 
            exclusive_raws={}, 
            exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
            )
        return ds, img_transformer, target_transformer, im_target_trans
    else:
        ds = KTTIDataset(
            path='/share/data/KTTI/training/label_02', 
            img_path='/share/data/KTTI/trackong_image/training/image_02', 
            exclusive_raws={}, 
            exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
            img_transformer=img_transformer,
            target_transformer=target_transformer,
            im_target_trans=im_target_trans
            )
        ds.type_id = ['Car', 'Pedestrian']
        return ds

def build_KTTIFragmentDataset(label_path, img_path):
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
    ds = KTTIFragmentDataset(
        path=label_path, 
        img_path=img_path, 
        len=3,
        exclusive_raws={'track_id':[-1], 'type':['Misc', 'Cyclist', 'Person', 'Tram', 'Truck', 'Van']}, 
        exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
        img_transformer=img_transformer,
        target_transformer=target_transformer,
        im_target_trans=im_target_trans
    )
    return ds

# %%
if __name__ =='__main__':
    import matplotlib.pyplot as plt
    # ds = build_KTTIDataset(partial={'behind': 0.7})
    ds = build_KTTIFragmentDataset(
        label_path=Path(r"E:\KTTI\training\label_02"),
        img_path=Path(r"E:\KTTI\trackong_image\training\image_02")
    )
    # train_dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_to_list_fragment)
#%%

# def show_imgs(imgs, targets):
#     fig, axs = plt.subplots(1, 3, sharex='col', sharey='row')
#     axs = axs.flatten()
#     for idx, img in enumerate(imgs):
#         axs[idx].imshow(img)
#     fig.show()
# ds.register_imgs_hook(show_imgs)
# imgs, targes = next(iter(train_dl))
