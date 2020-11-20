#%%
import torch
import torch.nn.functional as F
import torchvision
import easydict
from torch import nn

from backbone import random_input, build_backbone
from transformer import DETRTransformer, build_DETR_transformer
#%%
args = easydict.EasyDict({'lr_backbone': 1, 'masks':False, 'position_embedding': 'v3',
                        'backbone': 'resnet50', 'dilation': True, 'hidden_dim': 256})
model = build_backbone(args)
t_args = easydict.EasyDict({
    'd_model': 256,
    'dropout': 0.1,
    'nhead': 8,
    'dim_feedforward':2048,
        # num_encoder_layers=args.enc_layers,
        # num_decoder_layers=args.dec_layers,
        # normalize_before=args.pre_norm,
    'return_intermediate_dec': True,
})
transformer = build_DETR_transformer(t_args)
#%%
def temporary_model(x):
    features, pos = model(x)
    src, mask = features[-1].decompose()
    assert mask is not None
    input_proj = nn.Conv2d(model.num_channels, 256, kernel_size=1)
    query_embed = nn.Embedding(100, 256)
    hs = transformer(input_proj(src), mask, query_embed.weight, pos[-1])[0]
    return hs
#%%
import sys
sys.path.append('...')
# from ..utils.datatranformer import 
from torch.utils.data import DataLoader, Dataset
from py.dataprocessing.CoCoDataset import CocoDataset
import misc
from torchvision import transforms

anns_file = "/share/data/annotations/instances_train2017.json"
image_path = "/share/data/train2017"
image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
ds = CocoDataset(anns_file, image_path, image_transform=image_transformer)
dl = DataLoader(ds, collate_fn=misc.collate_fn)
#%%
for srcs, targets in dl:
    output = temporary_model(srcs)
    break

#%%
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
model = torch.nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(1, 1, bias=False)),
    ('linear2', nn.Linear(1, 1, bias=False))
]))
for param in model.parameters():
    nn.init.constant_(param, 1)
model = torchvision.models._utils.IntermediateLayerGetter(
    model,
    return_layers={'linear1': 'feature1', 'linear2': 'feature2'}
)
#%%
out = model(torch.ones(1) * 2)
y = out['feature1'] + out['feature2']
model.zero_grad()
print([param.grad for param in model.parameters()])
y.backward()
print([param.grad for param in model.parameters()])

