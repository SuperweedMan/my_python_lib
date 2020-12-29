# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .transformer import build_transformer

def build_model(args):
    return build(args)
