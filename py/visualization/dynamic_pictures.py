#%%
# %matplotlib inline
import numpy as np
import os
os.environ["DISPLAY"]=":1"
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
# from matplotlib import animation, rc
# from IPython.display import HTML

fig, ax = plt.subplots(dpi=72, figsize=(8,6))

x = np.arange(-2*np.pi, 2*np.pi, 0.01)
y = np.sin(x)

line,  = ax.plot(x, y)

def init():
    ax.plot(x, y)
    # line.set_ydata(np.sin(x))
    # return line

def animate(i):
    plt.cla()
    ax.plot(x, np.sin(x+i/10.0))
    # line.set_ydata(np.sin(x+i/10.0))
    # return line

anim = animation.FuncAnimation(fig=fig, 
                                       func=animate,
                                       frames=100, # total 100 frames
                                       init_func=init,
                                       interval=20,# 20 frames per second
                                       blit=False)
# anim.save('sinx.gif', writer='imagemagick')
plt.show()