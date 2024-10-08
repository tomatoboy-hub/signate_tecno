import logging 
import os
import wandb
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    EarlyStopping,
)

from pytorch_lightning.loggers import WandbLogger

from src.datamodule.datamodule import TECNODataModule
from src.modelmodule.modelmodule import TECNOModel

logging.basicConfig(level = logging.INFO,format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)

@hydra.main(config_path = "conf", config_name="train", version_base="1.3")
def main(cfg:DictConfig):
    # seed_everything(cfg.seed)
    directory_path = f"/{cfg.base_path}/output"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode = cfg.monitor_mode,
        save_top_k=1,
        save_last = False
    )

    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    wandb.init(
        dir = directory_path,
        name = cfg.exp_name,
        project = cfg.project,
        entity = cfg.wandb_entity,
    )

    pl_logger = WandbLogger(
        name = cfg.exp_name,
        project=cfg.project,
        entity = cfg.wandb_entity,
        save_dir=f"{directory_path}/wandb_logs"
    )
    
    datamodule = TECNODataModule(cfg)
    LOGGER.info("Set Up DataModule")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = TECNOModel(cfg)

    trainer = Trainer(
        #env
        default_root_dir = Path.cwd(),
        #num_nodes = cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        #trainning
        fast_dev_run=cfg.debug,
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloder()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb,lr_monitor,model_summary],
        logger=pl_logger,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

