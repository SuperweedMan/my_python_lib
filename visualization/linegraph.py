#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def plot(lines, initial_x=0, smooth_arg=10, alpha=0.3):
    '''
    输入：
        lins: 每个曲线可以是list或者dict类型，存储其y值。其x值默认为0，可指定。
              对于用dict类型，key为标识，value为存储y值的list。
    '''
    assert isinstance(lines, (dict, list))  # 类型判断, 输入的必须是list或者dict
    token_place = OrderedDict()
    if isinstance(lines, dict):  # dict展开成list，并且存储标识
        for k, v in lines.items():
            assert isinstance(v, list)  # dict里面的value，必须是list类型，其元素是y值
            token_place[k] = len(v)  # 记录标识名与位置
        token_place = {k: v + sum(list(token_place.values())[:idx]) for idx, (k, v) in enumerate(token_place.items())}
        lines = [y for k, v in lines.items() for y in v]  # 展开为list
    
    x = np.arange(initial_x, len(lines))
    y = np.array(lines)
    y_smooth = np.convolve(y, np.ones((smooth_arg,))/smooth_arg, mode='same')

    # plt.figure(figsize=(18, 6))  # 设置画布大小，单位100像素
    plt.plot(x, y, ls='-', lw=2, label='plot figure',  alpha=alpha, c='darkorange')
    plt.plot(x, y_smooth, ls='-', lw=2, label='plot figure',  alpha=1, c='darkorange')
    # plt.xlim(1, x.shape[0])  
    # plt.xlim(1, 50)

    for k, v in token_place.items():
        plt.axvline(x=v, c='black', ls='--', lw=2)

    plt.show()

#%%
if __name__ == '__main__':
    y = {'0':[1,2,3]*10, '1':[1,2,3] * 10, '2':[1,2,3] * 10}
    plot(y)