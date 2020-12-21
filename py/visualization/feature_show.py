#%%
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from py.visualization.plt_operations import AxesOperations
from torch._C import HOIST_CONV_PACKED_PARAMS
from typing_extensions import runtime

#%%
class FeatureMap:
    def __init__(self):
        self.features = {}
    
    def get_features(self, name):
        assert isinstance(name, str)
        def hook(module, input, output):
            self.features[name] = output.detach().clone()
        return hook

    def heatmap(self, name: str, ax: matplotlib.axes._axes.Axes):
        assert isinstance(name, str)
        if name in self.features:
            axes_operator = AxesOperations(ax)
            axes_operator.heatmap(self.features[name].numpy())
        else:
            raise RuntimeError("Can't find feature mpa named {}".format(name))

class AttentionMap(FeatureMap):
    def __init__(self):
        super(AttentionMap, self).__init__()

    def get_features(self, name):
        assert isinstance(name, str)
        def hook(module, input, output):
            self.features[name] = output.detach().clone()
        return hook
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