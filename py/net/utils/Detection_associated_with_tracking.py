#%%
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch
import numpy as np
from torch.nn.modules import linear
import torchvision
from torchvision.ops import box_convert
#%%
class masked_tracking:
    def __init__(self) -> None:
        self.storage_list = []
        self.mask = torch.tensor([], dtype=torch.bool)

    def append(self, data):
        self.storage_list.append(data)
        self.mask = torch.cat([self.mask, torch.tensor([True])]) 

    def get_availale(self):
        result = []
        idxs = torch.where(self.mask)[0]  # 使能的地方
        for idx in idxs:
            result.append(self.storage_list[idx])
        return result

    def disable_all(self):
        self.mask = torch.zeros_like(self.mask).bool()
    
    def any_available(self):
        return bool(self.mask.any())
    
    def update_storage_data(self, idx, data):
        idxs = torch.where(self.mask)[0]  # 使能的地方
        self.storage_list[idxs[idx]] = data  # 从新的idx 转到 内部 idx

    def get_storage_idx(self, idxs):
        assert isinstance(idxs, (list, torch.Tensor, np.ndarray))  # idx 是输出的idx
        if isinstance(idxs, (list, np.ndarray)):
            idxs = torch.tensor(idxs)
        mask_idxs = torch.where(self.mask)[0]  # mask里面为True的idx
        idxs = torch.index_select(mask_idxs, dim=0, index=idxs)
        return idxs

    def update_mask(self, idxs):
        if len(idxs) != 0:
            assert isinstance(idxs, (list, torch.Tensor, np.ndarray))  # idx 是输出的idx
            if isinstance(idxs, (list, np.ndarray)):
                idxs = torch.tensor(idxs)
            mask_idxs = torch.where(self.mask)[0]  # mask里面为True的idx
            idxs = torch.index_select(mask_idxs, dim=0, index=idxs)
            self.mask = torch.index_fill(self.mask, dim=0, index=idxs, value=False)

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
    def __init__(self, max_distance: float, cost_class: float=1, cost_bbox: float=1, cost_giou:float=1) -> None:
        """初始化，并设定各个判定标准的权重"""
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.max_distance = max_distance
        # self.tracking_list = [{
        #         'labels': torch.tensor([1, 0]).to('cuda'), 
        #         'boxes': torch.tensor([[0.1236, 0.7684, 0.2472, 0.4578],
        #                                 [0.5974, 0.5812, 0.0537, 0.2921]]).to('cuda')
        #     }]  # test
        self.tracking_list = masked_tracking()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs):
        """
        进行匹配， 每次只能匹配一张图片
        输入为本次检测结果
        输出为 c:代价矩阵
             row_ind: 本次检测box的index
             col_ind: 
        """
        assert outputs['pred_logits'].shape[0] == 1  # 一次只能处理一张图片
        # 检测目标为空
        if outputs['pred_logits'].numel() ==0:
            self.tracking_list.disable_all()  # 意味着所有的tracking没了，需要重新编号
        # 跟踪目标为空
        elif not self.tracking_list.any_available():  # 没有正在跟踪的目标
            bs, num_queries = outputs["pred_logits"].shape[:2]
            for idx in range(num_queries):
                self.tracking_list.append({
                    'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][idx])]).detach().clone(),
                    'boxes': outputs['pred_boxes'][0][idx].detach().clone()
                })
        else:  
            # 检测目标跟跟踪目标都不为空
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in self.tracking_list.get_availale()])
            tgt_bbox = torch.stack([v["boxes"] for v in self.tracking_list.get_availale()])

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
            
            c = C[0].numpy()
            row_ind, col_ind = linear_sum_assignment(c)
            trace_id = self.tracking_list.get_storage_idx(col_ind)  # 必须在update_mask之前
                        
            # 更新前先读取，防止前面的变化影响后面的
            temp = self.tracking_list.get_availale()
            update_mask_list = []  # 用来存储需要mask的，最后一起mask
            #遍历匹配到的，超过阈值的也成为匹配失败
            for row, col in zip(row_ind, col_ind):
                if c[row, col] > self.max_distance:
                    self.tracking_list.append({
                        'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][row])]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
                    update_mask_list.append(col)
                else:
                    # 匹配成功的，更新数据
                    self.tracking_list.update_storage_data(col, {
                        'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][row])]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
                        
            # 未匹配到的，加入为新的跟踪目标
            for row, detection in enumerate(outputs['pred_logits'][0]):
                if row not in row_ind:  # 找不到匹配的
                    self.tracking_list.append({
                        'labels': torch.tensor([torch.argmax(detection)]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
            # 为匹配到的，变为不活动跟踪目标
            for col in range(len(temp)):
                if col not in col_ind:
                    update_mask_list.append(col)
            self.tracking_list.update_mask(update_mask_list)
            return c, row_ind, trace_id