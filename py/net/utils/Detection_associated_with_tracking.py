#%%
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch
import numpy as np
import torchvision
from torchvision.ops import box_convert
#%%
a_cost = np.array([
    [4, 1, 3],
    [2, 0, 5]
])
b_cost = np.array([
    [4, 1],
    [2, 0],
    [3, 2]
])
row_idx, col_idx = linear_sum_assignment(b_cost)
print(row_idx, col_idx)
print(b_cost[row_idx, col_idx])
#%%
class tracking_matcher(nn.Module):
    """
    用于匹配检测框到已有的跟踪目标上
    fram0：
        所有检测到的目标都新增到跟踪目标集合
    frame1：
        所有检测到的目标与跟踪目标匹配
        没有匹配到检测目标的加到跟踪目标集合
        没有匹配到的跟踪目标移出跟踪目标集合
    """
    def __init__(self, cost_class: float=1, cost_bbox: float=1, cost_giou:float=1) -> None:
        """初始化，并设定各个判定标准的权重"""
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        # self.tracking_list = [{
        #         'labels': torch.tensor([1, 0]).to('cuda'), 
        #         'boxes': torch.tensor([[0.1236, 0.7684, 0.2472, 0.4578],
        #                                 [0.5974, 0.5812, 0.0537, 0.2921]]).to('cuda')
        #     }]  # test
        self.tracking_list=[]
        self.tracking_mask_idx = 0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs):
        """
        进行匹配， 每次只能匹配一张图片
        """
        assert outputs['pred_logits'].shape[0] == 1  # 一次只能处理一张图片
        # 检测目标为空
        if outputs['pred_logits'].numel() ==0:
            self.tracking_mask_idx = len(self.tracking_list)  # 意味着所有的tracking没了，需要重新编号
        # 跟踪目标为空
        elif len(self.tracking_list) == self.tracking_mask_idx:  # 没有正在跟踪的目标
            bs, num_queries = outputs["pred_logits"].shape[:2]
            for idx in range(num_queries):
                self.tracking_list.append({
                    'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][idx])]),
                    'boxes': outputs['pred_boxes'][0][idx]
                })
        else:  
            # 检测目标跟跟踪目标都不为空
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in self.tracking_list])
            tgt_bbox = torch.stack([v["boxes"] for v in self.tracking_list])

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = 1-out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = 1-torchvision.ops.generalized_box_iou(
                    box_convert(out_bbox, 'cxcywh', 'xyxy'),
                    box_convert(tgt_bbox, 'cxcywh', 'xyxy') 
                )
            # 代价矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()
            