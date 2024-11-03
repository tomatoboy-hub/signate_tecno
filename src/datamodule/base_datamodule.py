from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule  
from torchvision import transforms as T
from PIL import Image
import cv2
import torch
import os
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
        
class CustomDataModule(LightningDataModule):
    def __init__(self,train_df,image_transform,batch_size,fold):
        super().__init__()
        self.fold = fold
        self.train = train_df
        self.transform = image_transform
        self.batch_size = batch_size
        
    def setup(self):
        df_train = self.train[self.train.kfold != self.fold].reset_index(drop= True)
        df_valid = self.train[self.train.kfold != self.fold].reset_index(drop = True)

        self.train_dataset = CustomDataset(df_train,self.transform, data_type = "train")
        self.valid_dataset = CustomDataset(df_valid,self.transform, data_type = "train")

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last  =True
        )
        return train_loader
    
    def valid_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size = self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )
        return valid_loader

        
