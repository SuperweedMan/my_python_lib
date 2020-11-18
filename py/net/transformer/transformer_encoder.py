#%%
import torch
import torch.nn as nn
import copy

#%%
def clones(module, num):
    """
    Copy a module several times.
    intput:
        module: module need to be copyed.
        num: copy times.
    return:
        return a nn.ModuleList container.
    """
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(num)]
    )

# %%
class StackedEncoder(nn.Module):
    """
    The encoder be consist of several submodules.
    """
    def __init__(self, submodule, num):
        super(StackedEncoder, self).__init__()
        # self.submodules (nn.ModuleList) 
        self.submodules = clones(submodule, num)
        self.norm = nn.LayerNorm()
    
    def forward(self, x, mask):
        for module in self.submodules:
            x = module(x, mask)
        
        return self.norm(x)