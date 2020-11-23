#%%
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import Tensor
from collections import Iterable

# %%
def boxes_percentage_to_value(boxes: Tensor, size):
    """
    Used to tranfer boxes from percentage to value (according to the size).
    **only** use for form (xyxy) and (cxcywh)
    inputs:
        boxes: tensor of boxes, shape is [N, 4]
        size: Iterable(int, int), size of img(H, W).
    """
    assert isinstance(boxes, Tensor)
    assert isinstance(size, Iterable)
    assert (len(boxes.shape) == 2) and (boxes.shape[-1] == 4)
    return torch.stack(
                        [line * size[1] if index % 2 == 0 else line * size[0] for index, line in enumerate(boxes.T)], 
                        dim=-1)
#%%
if __name__ == "__main__":
    boxes = torch.ones(7, 4 )
    print(boxes_percentage_to_value(boxes, (500, 600)))