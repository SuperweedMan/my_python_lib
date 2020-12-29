#%%
from typing import overload
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from py.visualization.plt_operations import AxesOperations
from torch._C import HOIST_CONV_PACKED_PARAMS
from typing_extensions import runtime
from collections import Iterable
from typing import Tuple, List

#%%
class FeatureMap:
    def __init__(self):
        self.features = {}
        self.temp_name = None
        self.handles = {}
    
    def hook_function(self, name):
        assert isinstance(name, str)
        self.temp_name = name
        def hook(module, input, output):
            self.features[name] = output.detach().clone()
        return hook

    def keep_handle(self, handle: torch.utils.hooks.RemovableHandle):
            self.handles[self.temp_name] = handle

    def remove_hooks(self, name):
        assert isinstance(name, str)
        if name in self.features:
            self.handles[name]()  # 调用以解除hook
        else:
            raise RuntimeError("Can't find feature map named {}".format(name))

    def get_feature(self, name):
        assert isinstance(name, str)
        if name in self.features:
            return self.features[name]
        else:
            raise RuntimeError("Can't find feature map named {}".format(name))

    def heatmap(self, name: str, ax: matplotlib.axes._axes.Axes, index: int=0):
        assert isinstance(name, str)
        if name in self.features:
            axes_operator = AxesOperations(ax)
            if isinstance(self.features[name], torch.Tensor):
                axes_operator.heatmap(self.features[name].cpu().numpy())
            else:
                axes_operator.heatmap(self.features[name][index].cpu().numpy())
        else:
            raise RuntimeError("Can't find feature map named {}".format(name))

class AttentionMap(FeatureMap):
    """作为钩子函数读取注意力权重，注意：tensor 经过 .detach().clone()"""
    def __init__(self):
        super(AttentionMap, self).__init__()

    def hook_function(self, name):
        assert isinstance(name, str)
        self.temp_name = name
        def hook(module, input, output):
            self.features[name] = output[1].detach().clone()
        return hook
    
    def heatmap(self, name: str, ax: matplotlib.axes._axes.Axes, bs: int, indexs: Tuple[int,int], img_shape: Tuple[int, int], **kwargs):
        assert name in self.features 
        assert isinstance(indexs, Iterable)
        axes_operator = AxesOperations(ax)
        img_tensor = make_grid(self.features[name][bs][indexs[0]:indexs[1]].cpu().reshape(indexs[1]-indexs[0], 1, *img_shape), **kwargs)
        img = torch.sum(img_tensor, dim=0, keepdim=True).numpy().transpose(1, 2, 0)
        axes_operator.heatmap(img)
# %%
if __name__ == "__main__":
    linear = nn.Linear(10, 10)
    feature_hook = FeatureMap()
    linear.register_forward_hook(feature_hook.get_features('number_1'))
    fig, ax = plt.subplots(1, 1)
    x = torch.rand(1, 10)
    y = linear(x)
    feature_hook.heatmap('number_1', ax)
    fig.show()