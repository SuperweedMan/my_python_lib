#%%
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from matplotlib import animation
import re
from py.visualization.plt_operations import AxesOperations
from py.visualization import box_ops
from py.dataprocessing import KTTIdataset
import  torch

#%%
imgs_path = '/share/data/KTTI/trackong_image/training/image_02'
predicted_path = '/share/data/KTTI_predictions/'
img_size = (370, 1224)

predicted_label_files = os.listdir(predicted_path)
predicted_label_files.sort()
nums = [re.findall(r"\d+",name) for name in predicted_label_files]
fragment_num, _ = zip(*nums)
fragment_nums = [fragment_num.count(num) for num in ['{:0>4d}'.format(i) for i in range(21)]]

class MyAnimate:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(12.24, 3.7))
        ax = self.fig.add_subplot(1,1,1)
        self.ax_operator = AxesOperations(ax)
    
    def initial(self):
        pass

    def named_animate(self, fragment):
        fragment = fragment
        def animate(frame):
            # 清空
            self.ax_operator.return_axes().cla()
            # 获取预测信息
            file_name = 'fragment_{:0>4d}-frame{:0>6d}.json'.format(fragment, frame)
            print(file_name)
            if os.path.exists(os.path.join(predicted_path, file_name)):
                with open(os.path.join(predicted_path, file_name), 'r') as f:
                    predicted_data = json.load(f)
                    predicted_data = {k: torch.tensor(v) for k, v in predicted_data.items()}

                if predicted_data['pred_boxes'].numel() != 0:
                    self.ax_operator.labeled_boxs(
                            predicted_data, 
                            'cxcywh', 
                            ['Car', 'Pedestrain'], 
                            torch.tensor(img_size),
                        )
            # 读取图像
            if os.path.exists(
                os.path.join(
                    imgs_path, '{:0>4d}'.format(fragment), '{:0>6d}.png'.format(frame)
                )
            ):
                img = Image.open(
                    os.path.join(
                        imgs_path, '{:0>4d}'.format(fragment), '{:0>6d}.png'.format(frame)
                    )
                )
                self.ax_operator.return_axes().imshow(img)
        return animate

for fragment in range(21):
    my_anim = MyAnimate()
    anim = animation.FuncAnimation(fig=my_anim.fig, 
                                    func=my_anim.named_animate(fragment),
                                    frames=fragment_nums[fragment], # total 100 frames
                                    init_func=my_anim.initial,
                                    interval=5,# 20 frames per second
                                    blit=False)
    anim.save('fragment_{:0>4d}.gif'.format(fragment), writer='imagemagick')

# %%
