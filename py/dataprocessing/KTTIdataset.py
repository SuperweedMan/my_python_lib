#%%
# -----------------------------------------------------------------
# classes:
#    KTTIdataset: 根据输入参数创建一个KTTI的dateset实例。
#    CropAnns: 删除不需要的categories。
# functions:
#    collate_to_list: dataloader组合函数。
# -----------------------------------------------------------------
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

#%%
def get_targets_from_files(targets_path, exclusive_raws={}, exclusive_cats=[]):
    if not os.path.exists(targets_path):
        raise ValueError('wrong path')
    all_targets = {}
    label_names = os.listdir(targets_path)
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

#%%
class KTTIDataset(Dataset):
    """
    创建一个KTTI dataset
    """
    type_id =  ['Car', 'Cyclist', 'Misc', 'Pedestrian', 'Person', 'Tram', 'Truck', 'Van'] 
    def __init__(self, path, img_path, return_frame=True, exclusive_raws={}, exclusive_cats=[], img_transformer=None, target_transformer=None):
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
        all_targets = get_targets_from_files(path, exclusive_raws=exclusive_raws, exclusive_cats=exclusive_cats)  # 返回的是pandas格式
        self.all_targets_dict = {}
        self.return_frame = return_frame
        self.imgs_path = img_path
        self.img_transformer = img_transformer
        self.target_transformer = target_transformer
        if 'type' in exclusive_raws:
            for item in exclusive_raws['type']:
                if item in type_id:
                    type_id.remove(item)
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
                label_dict['labels'] = [type_id.index(t) for t in label_dict['type']]
                targets.append(label_dict)
            self.all_targets_dict[fragment] = targets
        self.num = [len(v) for k, v in self.all_targets_dict.items()]
        if return_frame is True:
            self.all_targets_dict = [v for k, v in self.all_targets_dict.items()]
            self.all_targets_dict = [x for l in self.all_targets_dict for x in l]
    
    def type_id(self):
        """
        return a list contains sorted types.
        """
        return type_id

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
            target = self.all_targets_dict[idx]
            img_path = os.path.join(self.imgs_path, target['fragment'], '{:0>6d}'.format(target['frame']) + '.png')
            img = Image.open(img_path)
            if self.target_transformer is not None:
                target = self.target_transformer(target)
            if self.img_transformer is not None:
                img = self.img_transformer(img)
        else:
            raise ValueError('The function to return a entire sequence is not yet implemented.')
        return img, target

    def __len__(self):
        return sum(self.num)

def collate_to_list(batch):
    return list(zip(*batch))

# class AnnsToTensor:
#     '''
#     将values转换为tensor
#     '''
#     def __init__(self, name=[]):
#         self.name_list = name

#     def __call__(self, anns):
#         targets = {}
#         for name in anns.keys():
#             if name in self.name_list:
#                 targets[name] = torch.tensor(anns[name])
#             else:
#                 targets[name] = anns[name]
#         return targets

class CropAnns:
    def __init__(self, name=[]):
        self.name = name
    
    def __call__(self, anns):
        target = {}
        for name in anns:
            if name in self.name:
                target[name] = anns[name]
        return target
# %%
if __name__ =='__main__':
    img_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),       
    ])
    target_transformer = transforms.Compose([
        CropAnns(['boxes', 'labels'])
    ])
    
    ds = KTTIDataset(
        path='/share/data/KTTI/training/label_02', 
        img_path='/share/data/KTTI/trackong_image/training/image_02', 
        exclusive_raws={'track_id':[-1], 'type':['Misc', 'Cyclist', 'Person', 'Tram', 'Truck', 'Van']}, 
        exclusive_cats=['dimensions_0', 'dimensions_1', 'dimensions_2'],
        img_transformer=img_transformer,
        target_transformer=target_transformer
        )
    train_dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_to_list)