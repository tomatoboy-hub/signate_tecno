#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
classifications = ["not-hold a folding fan", "hold a folding fan"]

def search_images(directory):
    # 対応する画像の拡張子を定義
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    # 画像ファイルのパスとファイル名をリストに格納
    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                image_files.append({
                    'file_name': file,
                    'file_path': file_path
                })

    # pandas DataFrame に変換
    df = pd.DataFrame(image_files)

    return df

#HOME = Path("/content")
#INPUTS = HOME / "dataset"  # input data
INPUTS = Path("/root/signate_tecno/input")
TRAIN_IMAGEDIR0 = INPUTS / "train" / "not-hold"
TRAIN_IMAGEDIR1 = INPUTS / "train" / "hold"

train_df = pd.DataFrame()
train_df0 = search_images(TRAIN_IMAGEDIR0)
train_df1 = search_images(TRAIN_IMAGEDIR1)
train_df0['caption'] = classifications[0]
train_df1['caption'] = classifications[1]
train_df0['label'] = 0
train_df1['label'] = 1
train_df = pd.concat([train_df0, train_df1], axis=0)
print(train_df)
train_df.to_csv('/root/signate_tecno/input/train.csv', index=None)


# In[4]:


import os, shutil

#必要なライブラリのインポート
import re, gc, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import warnings, random
import cv2

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.cuda.amp import GradScaler

import timm
import yaml
from tqdm import tqdm
import time
import copy
from collections import defaultdict

from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL


# In[5]:


ARGS = {
  'DATA_DIR': '/root/signate_tecno/input/',
  'OUT_DIR': '/root/signate_tecno/output',
  'model_name': 'vit_l_16',
  'image_size': (224, 224), # vit_l16
  #cpu(slow)
  #'train_batch_size': 4,
  #'test_batch_size': 8,
  #gpu
  #'train_batch_size': 28, # 32(x)
  #'test_batch_size': 56,
  #'n_fold': 2,
  #'epochs': 3,
  #'timm_model_name': 'resnet50',
  #'timm_model_name': 'vit_base_patch16_224',
  #lb:0.8545424 ? :
  #'timm_model_name': 'vit_large_patch32_224_in21k',
  #down?
  #'timm_model_name': 'vit_huge_patch14_224_in21k',
  #colab free crash memory?.(batch=4)
  #'timm_model_name': 'vit_giant_patch14_224_clip_laion2b',
  #Only one class present in y_true. ROC AUC score is not defined in that case.
  #lb:0.8944379  : top, batchsize : 4, 'image_size': (448, 448),
  'timm_model_name': "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
  'pretrained': True,
  'n_fold': 3, # 5
  'epochs': 4, # 8
  'image_size': (384, 384), # eva02_large_patch14_448
  'criterion': 'CrossEntropy',
  #'is_blurry': True,
  'is_blurry': False,
  #'image_size': (336, 336),
  #GPU: 16GB
  'train_batch_size': 5, # 4, #1(ng?)
  'test_batch_size': 15, #x3?
  #GPU: 80GB?
  #'train_batch_size': 32, # 4, #1(ng?)
  #'test_batch_size': 96,
  #batchsize row is roc_auc calc ng,(batchsizeが小さいと、class data が片方のみ存在している状態になり roc_auc の計算ができない)
  'seed': 2023,
  'optimizer': 'AdamW',
  'learning_rate': 1e-05,
  'scheduler': 'CosineAnnealingLR', # CosineAnnealingWarmRestarts
  'min_lr': 1e-06,
  'T_max': 500,
  'n_accumulate': 1,
  'clip_grad_norm': 'None',
  'apex': True,
  'num_classes': 2,
  'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  }


# In[6]:


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler2)
    return logger

#再現性を出すために必要な関数となります
def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ['PYTHONHASHSEED'] = str(worker_id)

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


LOGGER = get_logger(ARGS['OUT_DIR']+'train')
set_seed(ARGS["seed"])


# In[15]:


def create_folds(data, num_splits, seed):
    data["kfold"] = -1

    mskf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    labels = ["label"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data

train = pd.read_csv(f"{ARGS['DATA_DIR']}/train.csv")
train = create_folds(train, num_splits=ARGS["n_fold"], seed=ARGS["seed"])
print("Folds created successfully")

train.head()


# In[7]:


class CustomDataset(Dataset):
    def __init__(self, df, transform, data_type):
        self.df = df
        self.data_type = data_type

        if self.data_type == "train":
            self.image_paths = df['file_path']
            self.labels = df['label']
        if self.data_type == "test":
            #self.image_paths = df[0]
            self.image_paths = df['image_path']

        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        if self.data_type == "train":
            #image = cv2.imread(f"/content/train/{file_name}")
            image = cv2.imread(f"{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)

            image = self.transform(image=image)["image"]
            return image, label

        if self.data_type == "test":
            if os.path.exists(f"/content/test/{image_path}"):
              image = cv2.imread(f"/content/test/{image_path}")
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              image = self.transform(image=image)["image"]
            else:
              print("not found " + image_path)
              image = None

            return image


# In[8]:


class CustomDataset2(Dataset):
    def __init__(self, df, transform, data_type, is_blurry):
        self.df = df
        self.data_type = data_type

        if self.data_type == "train":
            if is_blurry:
                self.image_paths = df['file_path']
            else:
                self.image_paths = df['file_path']
            self.labels = df['label']
        if self.data_type == "test":
            #self.image_paths = df[0]
            self.image_paths = df['image_path']

        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        if self.data_type == "train":
            image = cv2.imread(f"/content/train/{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)

            image = self.transform(image=image)["image"]
            return image, label

        if self.data_type == "test":
            image = cv2.imread(f"/content/test/{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform(image=image)["image"]

            return image


# In[9]:


import albumentations as albu
from albumentations.pytorch import transforms as AT

# Augumentation用
#CV : up, LB : down
image_transform_train = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.3),
    albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
    albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=45, p=0.5),
    albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AT.ToTensorV2()
])

image_transform = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    # albu.HorizontalFlip(p=0.5),
    # albu.VerticalFlip(p=0.5),
    # albu.RandomBrightnessContrast(p=0.3),
    # albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
    # albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=45, p=0.5),
    albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AT.ToTensorV2()
])


# In[10]:


def train_one_epoch(model, optimizer, train_loader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    running_score = []
    running_score_y = []
    scaler = GradScaler(enabled=ARGS["apex"])

    train_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, targets) in bar:
      images = images.to(device)
      targets = targets.to(device)

      batch_size = targets.size(0)
      with torch.cuda.amp.autocast(enabled=ARGS["apex"]):
          outputs = model(images)
          loss = criterion(ARGS, outputs, targets)

      scaler.scale(loss).backward()

      if ARGS["clip_grad_norm"] != "None":
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS["clip_grad_norm"])
      else:
          grad_norm = None

      scaler.step(optimizer)
      scaler.update()

      optimizer.zero_grad()

      if scheduler is not None:
          scheduler.step()

      train_loss.append(loss.item())

      running_loss += (loss.item() * batch_size)
      dataset_size += batch_size

      epoch_loss = running_loss / dataset_size

      running_score.append(outputs.detach().cpu().numpy())
      running_score_y.append(targets.detach().cpu().numpy())

      score = get_score(running_score_y, running_score)

      bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                      Train_Acc=score[0],
                      Train_Auc=score[1],
                      LR=optimizer.param_groups[0]['lr']
                      )
    gc.collect()
    return epoch_loss, score


# In[11]:


@torch.no_grad()
def valid_one_epoch(args, model, optimizer, valid_loader, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    preds = []
    valid_targets = []
    softmax = nn.Softmax()

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (images, targets) in enumerate(valid_loader):
      images = images.to(args["device"])
      targets = targets.to(args["device"])
      batch_size = targets.size(0)
      with torch.no_grad():
        outputs = model(images)
        predict = outputs.softmax(dim=1)
        loss = criterion(args, outputs, targets)

      running_loss += (loss.item() * batch_size)
      dataset_size += batch_size

      epoch_loss = running_loss / dataset_size

      preds.append(predict.detach().cpu().numpy())
      valid_targets.append(targets.detach().cpu().numpy())

      if len(set(np.concatenate(valid_targets))) == 1:
          continue
      score = get_score(valid_targets, preds)

      bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                      Valid_Acc=score[0],
                      Valid_Auc=score[1],
                      LR=optimizer.param_groups[0]['lr'])

    return epoch_loss, preds, valid_targets, score


# In[12]:


def one_fold(model, optimizer, schedulerr, device, num_epochs, fold):

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_score = np.inf
    best_prediction = None

    best_score = -np.inf
    for epoch in range(1, 1+num_epochs):
      train_epoch_loss, train_score = train_one_epoch(model, optimizer,
                                              train_loader=train_loader,
                                              device=device, epoch=epoch)

      train_acc, train_auc = train_score

      val_epoch_loss, predictions, valid_targets, valid_score = valid_one_epoch(ARGS,
                                                                                model,
                                                                                optimizer,
                                                                                valid_loader,
                                                                                epoch=epoch)
      valid_acc, valid_auc = valid_score

      LOGGER.info(f'Epoch {epoch} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {val_epoch_loss:.4f}')
      LOGGER.info(f'Epoch {epoch} - Train Acc: {train_acc:.4f}  Train Auc: {train_auc:.4f}  Valid Acc: {valid_acc:.4f}  Valid Auc: {valid_auc:.4f}')

      if valid_auc >= best_score:
        best_score = valid_auc

        print(f"{b_}Validation Score Improved ({best_epoch_score} ---> {valid_auc})")
        best_epoch_score = valid_auc
        best_model_wts = copy.deepcopy(model.state_dict())
        # PATH = f"Score-Fold-{fold}.bin"
        PATH = ARGS["OUT_DIR"] + f"Score-Fold-{fold}.bin"
        torch.save(model.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"Model Saved{sr_}")

        best_prediction = np.concatenate(predictions, axis=0)[:,1]

    end = time.time()
    time_elapsed = end - start

    LOGGER.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    LOGGER.info("Best Score: {:.4f}".format(best_epoch_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_prediction, valid_targets


# In[18]:


def create_model(args):
    model = models.vit_l_16(pretrained=True)
    model.heads[0] = torch.nn.Linear(in_features=model.heads[0].in_features, out_features=args["num_classes"], bias=True)
    return model

def create_model_timm(args):
    #model = timm.create_model(args["timm_model_name"], pretrained=True, num_classes=args["num_classes"])
    model = timm.create_model(args["timm_model_name"], args["pretrained"], num_classes=args["num_classes"])
    # 重みを更新するパラメータを選択する
    # 最終層だけでOK
    """
    params_to_update = []
    update_param_names = ['head.weight', 'head.bias']

    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    """
    return model

def criterion(args, outputs, labels, class_weights=None):
    if args['criterion'] == 'CrossEntropy':
      return nn.CrossEntropyLoss(weight=class_weights).to(args["device"])(outputs, labels)
    elif args['criterion'] == "None":
        return None

def fetch_optimizer(optimizer_parameters, lr, betas, optimizer_name="Adam"):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimizer_parameters, lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(optimizer_parameters, lr=lr, betas=betas)
    return optimizer

def fetch_scheduler(args, train_size, optimizer):

    if args['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args['T_max'],
                                                   eta_min=args['min_lr'])
    elif args['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=args['T_0'],
                                                             eta_min=args['min_lr'])
    elif args['scheduler'] == "None":
        scheduler = None

    return scheduler

def get_score(y_trues, y_preds):
    predict_list, targets_list = np.concatenate(y_preds, axis=0), np.concatenate(y_trues)
    predict_list_proba = predict_list.copy()[:, 1]
    predict_list = predict_list.argmax(axis=1)

    accuracy = accuracy_score(predict_list, targets_list)
    try:
        auc_score = roc_auc_score(targets_list, predict_list_proba)
    except:
        auc_score = 0.0

    return (accuracy, auc_score)

def prepare_loaders(args, train, image_transform, fold):
    df_train = train[train.kfold != fold].reset_index(drop=True)
    df_valid = train[train.kfold == fold].reset_index(drop=True)

    train_dataset = CustomDataset(df_train, image_transform, data_type="train")
    valid_dataset = CustomDataset(df_valid, image_transform, data_type="train")
    #train_dataset = CustomDataset2(df_train, image_transform, data_type="train", is_blurry=ARGS['is_blurry'])
    #valid_dataset = CustomDataset2(df_valid, image_transform, data_type="train", is_blurry=ARGS['is_blurry'])

    train_loader = DataLoader(train_dataset, batch_size=args['train_batch_size'],
                              worker_init_fn=worker_init_fn(args["seed"]),
                              num_workers=4,
                              shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args['test_batch_size'],
                              num_workers=4,
                              shuffle=False, pin_memory=True)

    return train_loader, valid_loader


# In[19]:


train_copy = train.copy()
LOGGER.info(ARGS)
for fold in range(0, ARGS['n_fold']):
    print(f"{y_}====== Fold: {fold} ======{sr_}")
    LOGGER.info(f"========== fold: {fold} training ==========")

    # Create Dataloaders
    train_loader, valid_loader = prepare_loaders(args=ARGS, train=train, image_transform=image_transform, fold=fold)
    #train_loader, valid_loader = prepare_loaders(args=ARGS, train=train, image_transform=image_transform_train, fold=fold)

    #vit_l16
    #model = create_model(ARGS)
    #heavy
    model = create_model_timm(ARGS)
    model = model.to(ARGS["device"])

    #損失関数・最適化関数の定義
    optimizer = fetch_optimizer(model.parameters(), optimizer_name=ARGS["optimizer"], lr=ARGS["learning_rate"], betas=(0.9, 0.999))

    scheduler = fetch_scheduler(args=ARGS, train_size=len(train_loader), optimizer=optimizer)

    model, predictions, targets = one_fold(model, optimizer, scheduler, device=ARGS["device"], num_epochs=ARGS["epochs"], fold=fold)

    print(predictions)
    train_copy.loc[train_copy[train_copy.kfold == fold].index, "oof"] = predictions
    #train_copy.loc[train_copy[train_copy.kfold == fold].index, "pred_0"] = predictions[:,0]
    #train_copy.loc[train_copy[train_copy.kfold == fold].index, "pred_1"] = predictions[:,1]

    del model, train_loader, valid_loader
    _ = gc.collect()
    torch.cuda.empty_cache()
    print()

scores = roc_auc_score(train_copy["label"].values, train_copy["oof"].values)
LOGGER.info(f"========== CV ==========")
LOGGER.info(f"CV: {scores:.4f}")


# In[ ]:


# OOF
#train_copy.to_csv(ARGS['OUT_DIR'] + f'oof.csv', index=False)
train_copy.to_csv(f"{ARGS['OUT_DIR']}/oof_CV{scores:.4f}.csv", index=False)
train_copy.to_csv(f"{ARGS['DATA_DIR']}/oof_CV{scores:.4f}.csv", index=False)


# In[ ]:


import os
os.environ['oof_CVSroce'] = f"{scores:.4f}"




# In[ ]:


#sample_submit.csvを読み込みます
#import pandas as pd
#submit = pd.read_csv(f"{ARGS['DATA_DIR']}/sample_submit.csv", header=None)
#submit = pd.read_csv(f"/content/drive/MyDrive/SIGNATE/Sense/sample_submit.csv", header=['image_path', 'label'])
#submit = pd.read_csv(f"/content/drive/MyDrive/SIGNATE/Sense/sample_submit.csv", header=None, names=['image_path', 'label'])
test = pd.read_csv(f"/root/signate_tecno/input/test.csv", header=None, names=['image_path'])
submit = pd.read_csv(f"/root/signate_tecno/input/sample_submit.csv", header=None, names=['image_path', 'label'])

test.head()
#submit.head()


# In[ ]:


# test用のデータ拡張
image_transform_test = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AT.ToTensorV2 ()
    ])
test_dataset = CustomDataset(submit, image_transform_test, data_type="test")
#test_dataset = CustomDataset2(submit, image_transform_test, data_type="test", is_blurry=ARGS['is_blurry'])
test_loader = DataLoader(test_dataset, batch_size=ARGS["test_batch_size"], shuffle=False, num_workers=1) # 4
#


# In[ ]:


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    predict_list = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, images in bar:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            #出力にソフトマックス関数を適用
            predicts = outputs.softmax(dim=1)

        predicts = predicts.cpu().detach().numpy()
        predict_list.append(predicts)
    predict_list = np.concatenate(predict_list, axis=0)
    #予測値が1である確率を提出します。
    predict_list = predict_list[:, 1]
    gc.collect()

    return predict_list


# In[ ]:


def inference(model_paths, dataloader, device):
    final_preds = []
    ARGS['pretrained'] = False
    for i, path in enumerate(model_paths):
        #model = create_model(ARGS)
        model = create_model_timm(ARGS)
        model = model.to(device)

        #学習済みモデルの読み込み
        model.load_state_dict(torch.load(path))
        model.eval()

        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


# In[ ]:


#!ls /content/test
print(f"{ARGS['OUT_DIR']}")
#!cp -rf ./Score-Fold-0.bin /content/drive/MyDrive/SIGNATE/Sense
#!cp -rf ./Score-Fold-1.bin /content/drive/MyDrive/SIGNATE/Sense


# In[ ]:


MODEL_PATHS = [
    f"{ARGS['OUT_DIR']}/Score-Fold-{i}.bin" for i in range(ARGS["n_fold"])
]


# In[ ]:


predict_list = inference(MODEL_PATHS, test_loader, ARGS["device"])


# In[ ]:


#submit['label'] = predict_list
submit['label'] = (predict_list > 0.5)
submit['label'] = submit['label'].astype(int)
submit.head()


# In[ ]:
submit.to_csv(f'{ARGS["DATA_DIR"]}/submission_CV{scores:.4f}.csv',index = False, header = None)




