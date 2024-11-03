from typing import Optional
import numpy as np
from pytorch_lightning import LightningModule
import torch 
import torch.nn as nn
import timm
import torch.optim as optim
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import get_cosine_schedule_with_warmup
import gc

class CustomModel(LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(cfg.model_name,pretrained=cfg.pretrained, num_classes = cfg.num_classes)
        params_to_update = []
        update_param_names = ['head.weight', 'head.bias']

        for name, param in self.model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False

    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.cfg.apex):
            outputs = self(images)
            loss = self.criterion(outputs, targets)

        # Backward pass and optimization step
        self.scaler.scale(loss).backward()
        
        if self.cfg.clip_grad_norm != "None":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

        self.scaler.step(self.optimizers())
        self.scaler.update()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Calculate accuracy and AUC (replace get_score with your custom function)
        score = self.__get_score(targets, outputs)
        self.log("train_acc", score[0], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auc", score[1], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Clean up memory
        gc.collect()
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # Softmaxで予測確率を取得
        preds = torch.softmax(outputs, dim=1)

        # AccuracyとAUCを計算
        score = self.__get_score(targets, preds)

        # 損失、精度、AUCをログする
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', score[0], prog_bar=True)
        self.log('val_auc', score[1], prog_bar=True)

        return {'val_loss': loss, 'val_acc': score[0], 'val_auc': score[1]}
    
    def __get_score(self,y_trues, y_preds):
        predict_list, targets_list = np.concatenate(y_preds, axis=0), np.concatenate(y_trues)
        predict_list_proba = predict_list.copy()[:, 1]
        predict_list = predict_list.argmax(axis=1)

        accuracy = accuracy_score(predict_list, targets_list)
        try:
            auc_score = roc_auc_score(targets_list, predict_list_proba)
        except:
            auc_score = 0.0

        return (accuracy, auc_score)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameter(),lr = self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps = self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    