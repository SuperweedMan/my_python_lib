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
        idx = len(self.storage_list)  # 长度刚好三本次append的位置的index
        self.storage_list.append(data)
        self.mask = torch.cat([self.mask, torch.tensor([True])])
        return idx

    def reset(self):
        self.storage_list = []
        self.mask = torch.tensor([], dtype=torch.bool)

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
        return int(idxs)

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

    def reset(self):
        self.tracking_list.reset()

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
            return (), torch.tensor([])  # 返回空的（idx-idx）和空的代价矩阵
        # 跟踪目标为空
        elif not self.tracking_list.any_available():  # 没有正在跟踪的目标
            bs, num_queries = outputs["pred_logits"].shape[:2]
            id_map = []
            for idx in range(num_queries):
                storage_idx = self.tracking_list.append({
                    'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][idx])]).detach().clone(),
                    'boxes': outputs['pred_boxes'][0][idx].detach().clone()
                })
                id_map.append((idx, storage_idx))
            return id_map, torch.zeros(num_queries, num_queries)
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
                        
            # 更新前先读取，防止前面的变化影响后面的
            temp = self.tracking_list.get_availale()
            update_mask_list = []  # 用来存储需要mask的，最后一起mask
            id_map = []  # tuple(检测目标idx, 跟踪目标track_id)
            #遍历匹配到的，超过阈值的也成为匹配失败
            for row, col in zip(row_ind, col_ind):
                if c[row, col] > self.max_distance:  # 代价过大，不与匹配
                    storage_idx = self.tracking_list.append({
                        'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][row])]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
                    update_mask_list.append(col)
                    id_map.append((row, storage_idx))
                else:
                    # 匹配成功的，更新数据
                    self.tracking_list.update_storage_data(col, {
                        'labels': torch.tensor([torch.argmax(outputs['pred_logits'][0][row])]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
                    id_map.append((row, self.tracking_list.get_storage_idx([col])))
                        
            # 未匹配到的检测目标，加入为新的跟踪目标
            for row, detection in enumerate(outputs['pred_logits'][0]):
                if row not in row_ind:  # 找不到匹配的
                    storage_idx = self.tracking_list.append({
                        'labels': torch.tensor([torch.argmax(detection)]).detach().clone(),
                        'boxes': outputs['pred_boxes'][0][row].detach().clone()
                    })
                    id_map.append((row, storage_idx))
            # 未匹配到的跟踪目标，变为不活动跟踪目标
            for col in range(len(temp)):
                if col not in col_ind:
                    update_mask_list.append(col)
            self.tracking_list.update_mask(update_mask_list)
            return sorted(id_map), c

#%%
if __name__ == "__main__":
    matcher = tracking_matcher(2.0)
    # 构造输入
    output_1 = {
        'pred_logits': torch.tensor([[[0.002, 0.998], [0.002, 0.998]]]),
        'pred_boxes': torch.tensor([[[0.1, 0.1, 0.001, 0.001], [0.5, 0.5, 0.002, 0.002]]])
    }
    output_2 = {
        'pred_logits': torch.Tensor(1, 0, 2),
        'pred_boxes': torch.Tensor(1, 0, 4)
    }
    output_3 = {
        'pred_logits': torch.tensor([[[0.002, 0.998], [0.002, 0.998], [0.002, 0.998]]]),
        'pred_boxes': torch.tensor([[[0.1, 0.1, 0.001, 0.001], [0.8, 0.8, 0.002, 0.002], [0.5, 0.5, 0.002, 0.002]]])
    }

    id_map_1, c_1 = matcher(output_1)  # 无跟踪目标有检测目标
    id_map_2, c_2 = matcher(output_3)  # 检测目标增加
    id_map_3, c_3 = matcher(output_1)  # 检测目标减少
    id_map_4, c_4 = matcher(output_2)  # 检测目标清空
    print(id_map_1, id_map_2, id_map_3, id_map_4)