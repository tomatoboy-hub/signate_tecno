import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import random
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
