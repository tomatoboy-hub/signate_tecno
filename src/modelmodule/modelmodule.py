from typing import Optional
import numpy as np
from pytorch_lightning import LightningModule
import torch 
import torch.nn as nn
import timm
import torch.optim as optim
from omegaconf import DictConfig
from transformers import get_cosine_schedule_with_warmup
from src.models.common import get_model

class TECNOModel(LightningModule):
    def __init__(self,cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.save_hyperparameters()
        self.training_step_loss = []
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf

    def forward(self,x:torch.Tensor, y: Optional[torch.Tensor]) -> dict[str,Optional[torch.tensor]]:
        return self.model(x,y)
    
    def training_step(self,batch,batch_idx):
        return self.__share_step(batch,'train')
    
    def validation_step(self,batch,batch_idx):
        return self.__share_step(batch,'val')
    
    def __share_step(self,batch, mode:str) -> torch.Tensor:
        x,y = batch
        output = self.model(x,y)
        loss: torch.Tensor = output["loss"]
        logits = output["logits"]

        if mode == 'val':
            self.validation_step_outputs.append(
                (
                    y.cpu().numpy(),
                    logits.detach().cpu().numpy(),
                    loss.detach().item(),
                )
            )
        self.log(
            f'{mode}_loss',
            loss.detach().item(),
            on_step=False,
            on_epoch = True,
            logger = True,
            prog_bar = True,
        )
        return loss
    
    def on_validation_epoch_end(self):
        y = np.concatenate([x[0] for x in self.validation_step_outputs])
        preds = np.concatenate([x[1] for x in self.validation_step_outputs])
        losses = np.array([x[2] for x in self.validation_step_outputs])
        loss = losses.mean()
        
        if loss < self.__best_loss:
            torch.save(self.model.state_dict(), f"best_model.pth")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_loss = loss
        self.validation_step_outputs.clear()

    def configure_optimziers(self):
        optimizer = optim.AdamW(self.model.parameters(),lr = self.cfg.optimzier.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps = self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]