from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule  
from torchvision import transforms as T
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self,cfg:DictConfig):
        self.cfg = cfg
        self.train_ids = cfg.split.train_study_ids
        self.img_size = cfg.img_size
        self.img_resize = cfg.img_resize
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p = 0.5),
            T.RandomAdjustSharpness(shapness_factor = 2, p = 0.5),
            T.Resize((self.img_resize ,self.img_resize)),
            T.ToTensor(),
            T.Normalize(mean = 0.5, std = 0.5)
        ])
    def __len__(self):
        return len(self.train_ids)
    
    def __getitem__(self,idx):
        image_path = self.train_ids[idx]['image_path']
        label = self.train_ids[idx]['label']

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return image, label

class ValidDataset(Dataset):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.valid_ids = self.cfg.valid_study_ids
        self.transform = T.Compose([
            T.Resize((cfg.image_resize, cfg.image_resize)),
            T.ToTensor(),
            T.Normalize(mean = 0.5, std = 0.5)
        ])
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self,idx):
        image_path = self.train_ids[idx]['image_path']
        label = self.train_ids[idx]['label']

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        return image, label

class TECNODataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        
    def train_dataloader(self):
        train_dataset = TrainDataset(cfg = self.cfg)
        train_loader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.batch_size,
            shuffle = True,
            num_workers = self.cfg.num_workers,
        )
        return train_loader
    
    def val_dataloader(self):
        valid_dataset = ValidDataset(cfg=self.cfg)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size = self.cfg.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
        )
        return valid_loader