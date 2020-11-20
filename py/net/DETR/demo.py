#%%
from PIL import Image

import requests
import matplotlib.pyplot as plt

from torch.nn import functional as F 
from torch import nn
from torchvision.models import resnet50
import torch
import torchvision.transforms as transforms

#%%
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, 
                    num_encoder_layer=6, num_decoder_layer=6, max_objs=100):
        """
        DETR demo
        """
        super(DETR, self).__init__()
        self.backbone = resnet50()
        in_features = self.backbone.fc.in_features
        del self.backbone.fc  # delete full connect layer.
        # using 1*1 conv change channels from in_features to hidden_dim
        self.conv = nn.Conv2d(in_features, hidden_dim, 1)
        # create a transformer
        self.transformer = nn.Transformer(
            hidden_dim, nhead=nheads, num_encoder_layers=num_encoder_layer,
            num_decoder_layers=num_decoder_layer
        )
        # note that the two linears below is reused in different obj.
        # classification layer, add backgroup category.
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # bbox(x, y, w, h)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # position embedding
        # output positon encoding
        self.query_pos = nn.Parameter(torch.rand(max_objs, hidden_dim))
        # spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propogate through resnet50 up to avg_pool
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        h = self.conv(x)  # transfor channels shape(n, c, h, w)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propogate through transformer
        # h (n, c, h ,w)->(n, c, h*w)->(h*w, n, c)
        # query_pos (max_objs, c)->(max_objs, 1, c)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                                self.query_pos.unsqueeze(1)
        ).transpose(0, 1)
        # h (output_len, n, c) -> (n, output_len, c)

        return {
            'pred_logits': self.linear_class(h),
            'pred_boxes': self.linear_bbox(h).sigmoid()
        }

#%%
model = DETR(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True
)
model.load_state_dict(state_dict)

#%%
model.eval()
img = Image.open('/share/data/train2017/000000581880.jpg')
transform = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]
    )
])
x = transform(img).unsqueeze(0)
output = model(x)
#%%
probas = output['pred_logits'].softmax(-1)[0,:,:-1]
keep = probas.max(-1).values > 0.7
boxes = output['pred_boxes'][0, keep] 

