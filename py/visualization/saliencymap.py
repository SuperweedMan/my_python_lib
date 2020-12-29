#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision

#%%
class SaliencyMap:
    def __init__(self, module):
        assert isinstance(module, nn.modules)  # 验证类型
        self.module = module


    def show(self, x, y):
        '''
        x是输入, y是标签，且二值化为(0, 1)
        '''
        assert isinstance(x, torch.Tensor)  # 输入量类型必须是Tensor
        assert isinstance(x, torch.Tensor)
        is_train_model = True if  module.training else False  # 记录模型的模式
        self.module.eval()  # 预测模式
        x.register_hook(self.__show_grad)
        out = torch.empty_like(self.module(x))  # 获取和输出一样size的tensor
        # out = out.gather(1, y.view(-1, 1)).squeeze() # 得到正确分类
        # logits.backward(torch.FloatTensor([1., 1., 1., 1., 1.])) # 只计算正确分类部分的loss
        out.data.resize_as(y).copy_(y)
        out.backward()
        if is_train_model:
            self.module.train()
    
    def __show_grad(grad):
        print(grad)

#%%
if __name__ == '__main__':
    # SaliencyMap()
    pass