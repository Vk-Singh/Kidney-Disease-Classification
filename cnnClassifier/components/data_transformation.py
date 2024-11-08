import os
from cnnClassifier.utils.logger import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from cnnClassifier.entity.config_entity import (DataTransformationConfig)

class DataSplit:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
       
        
    def split(self):
        df = pd.read_csv(self.config.root_dir + '/' +self.config.file)
        df_target = df["target"].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            df, df_target, test_size=self.config.test_data_size, random_state=int(os.environ['PYTHONHASHSEED']))

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=self.config.train_valid_split, random_state=int(os.environ['PYTHONHASHSEED']))
        
        return (X_train, y_train),(X_valid, y_valid),(X_test, y_test)

   
def transform(img_size):
    return {
        "train": A.Compose([
            A.Resize(img_size,img_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Downscale(p=0.25),
            A.ShiftScaleRotate(shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=60, 
                p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5    ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
                ),
                ToTensorV2()], p=1.),
    
        "valid": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            ToTensorV2()], p=1.)
            
        }   
        
class KidneyDataset(Dataset):
    def __init__(self, df, root_dir, transforms=None):
        self.df = df
        self.file_names = df['path'].values   
        self.targets = df['target'].values
        self.transforms = transforms
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img_path = self.root_dir + '/' + img_path
        #img_path = "/home/vikram/Downloads/pop_os_backup/Kidney-Disease-Classification-Deep-Learning-Project-main/artifacts/data_ingestion/" + img_path

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    
    
        
        