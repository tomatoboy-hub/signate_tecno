from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import timm

from src.utils.common import get_loss_fn

class TECNOTimm(nn.Module):
    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            features_only: bool,
            in_chans: int,
            n_classes: int,
            n_labels: int,
            loss_name: str,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name = model_name,
            pretrained = pretrained,
            features_only = features_only,
            in_chans = in_chans,
            num_classes = n_classes,
            global_pool = 'avg'
        )
        self.loss_fn = get_loss_fn(loss_name)
        self.n_labels = n_labels

    def forward(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits = self.model(x)
        output = {"logits": logits}
        if y is not None:
            loss = 0
            preds = self.loss_fn(logits,y)
        output["loss"] = loss
        return output