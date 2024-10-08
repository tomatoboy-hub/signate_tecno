from typing import Union
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.timm import TECNOTimm
MODELS = Union[TECNOTimm]

def get_model(cfg:DictConfig) -> MODELS:
    model:MODELS

    if cfg.model.name == "timm":
        model = TECNOTimm(
            cfg.model.model_name,
            cfg.model.pretrained,
            cfg.model.features_only,
            cfg.model.in_chans,
            cfg.model.n_classes, 
            cfg.model.n_classes, 
            cfg.model.n_label,
            cfg.loss_fn,
        )
    else:
        raise NotImplementedError
    
    return model