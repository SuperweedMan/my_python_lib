#%%
import matplotlib.pyplot as plt
import linegraph
import matplotlib
from typing import List, Dict

#%%
class AxesOperations:
    def __init__(self, ax):
        self.axes = ax
    
    def plot(self, lines: (List, Dict), initial_x=0, smooth_arg=10, alpha=0.3):
        linegraph.plot(self.axes, lines, initial_x, smooth_arg, alpha)

    def heatmap(self, data: np.ndarray):
        im = self.axes.imshow(data, cmap=plt.cm.hot_r)
        #增加右侧的颜色刻度条
        self.axes.colorbar(im)


    def return_axes():
        return self.axes

class FigOperations:
    def __init__(self, fig: matplotlib.figure.Figure=None)


# %%
if __name__ == "__main__":
    # 创建画布
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    axOperas = AxesOperations(ax1)
    # 画线
    y = {'0':[1,2,3]*10, '1':[1,2,3] * 10, '2':[1,2,3] * 10}
    axOperas.plot(y)
    # heatmap
    ax2 = fig.add_subplot(122)
    ax2Operas = AxesOperations(ax2)
    data = []
    for i in range(5):
        temp = []
        for j in range(5):
            k = np.random.randint(0,100)
            temp.append(k)
        data.append(temp)
    ax2Operas.heatmap(np.array(data))
    fig.show()