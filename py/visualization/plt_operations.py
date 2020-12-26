#%%
import matplotlib.pyplot as plt
from matplotlib.axes._base import _AxesBase
import numpy as np
import matplotlib
from py.visualization import linegraph
import matplotlib
import torch
import torchvision
from py.visualization import box_ops
from typing import List, Dict, Tuple

#%%
class AxesOperations:
    def __init__(self, ax: matplotlib.axes._axes.Axes):
        self.axes = ax
    
    def plot(self, lines: Tuple[List, Dict], initial_x=0, smooth_arg=10, alpha=0.3, **kargs):
        return linegraph.plot(self.axes, lines, initial_x, smooth_arg, alpha, **kargs)

    def heatmap(self, data: np.ndarray):
        im = self.axes.imshow(data, cmap=plt.cm.hot, interpolation='none')
        return im

    def labeled_boxs(self, data: Dict[torch.Tensor, torch.Tensor], box_form:str, 
        labels_str:str=[], box_percentage:Tuple[int, int]=None, additional_str:List[str]=[], edgecolor:str='green', linewidth:int=1, 
        fontsize:int=6, color:str='black', txt_box=dict(facecolor='g', joinstyle='round', alpha=0.6, lw=0, pad=0)):
        assert 'pred_logits' in data
        assert 'pred_boxes' in data
        assert box_form in ['cxcywh', 'xyxy', 'xywh']
<<<<<<< Updated upstream
        # data['pred_boxes'] = data['pred_boxes'][0]
=======
>>>>>>> Stashed changes
        if box_percentage is not None:  # 若输入的是百分比数据
            data['pred_boxes'] = box_ops.boxes_percentage_to_value(
                    data['pred_boxes'], box_percentage
                )
        data['pred_boxes'] = torchvision.ops.box_convert(
                data['pred_boxes'], box_form, "xywh"
            )  # boxes格式转换
        boxes = data['pred_boxes']
        for i, box in enumerate(boxes):  # 将数据画出
            rect = plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=edgecolor, linewidth=linewidth)
            self.axes.add_patch(rect)
            assert (len(data['pred_logits'][i]) == len(labels_str)) or len(labels_str) == 0 
            txt_str = "{}:{:.3f}".format(
                    labels_str[np.argmax(data['pred_logits'][i].detach().cpu().numpy())],
                    float(data['pred_logits'][i].max().detach().cpu().numpy())
                )
            self.axes.text(rect.xy[0], rect.xy[1], txt_str,
                    va='top', ha='left', fontsize=fontsize, color='black',
                    bbox=txt_box)
            if 'track_ids' in data:
                # data['track_ids'] = sorted(data['track_ids'])
                self.axes.text(rect.xy[0], rect.xy[1]+fontsize, str(data['track_ids'][i][1]),
                    va='top', ha='left', fontsize=fontsize, color='black',
                    bbox=txt_box)


    def return_axes(self):
        return self.axes
    

# %%
if __name__ == "__main__":
    # 创建画布
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # axOperas = AxesOperations(ax1)

    fig, ax = plt.subplots(2, 1)
    ax = ax.flatten()
    axOperas = AxesOperations(ax[0])
    # 画线
    y = {'0':[1,2,3]*10, '1':[1,2,3] * 10, '2':[1,2,3] * 10}
    axOperas.plot(y)
    # heatmap
    ax2Operas = AxesOperations(ax[1])
    data = []
    for i in range(5):
        temp = []
        for j in range(5):
            k = np.random.randint(0,100)
            temp.append(k)
        data.append(temp)
    
    fig.colorbar(ax2Operas.heatmap(np.array(data)), ax=ax[1])
    fig.show()