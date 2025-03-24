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
from pathlib import Path

from cnnClassifier.entity.config_entity import (DataTransformationConfig)


def transform(img_size):
    """
    Apply a series of augmentations to training and validation datasets.

    Parameters:
    img_size (int): The desired size to which the image will be resized.

    Returns:
    dict: A dictionary containing two keys, "train" and "valid", each mapped to
    an Albumentations.Compose object. The "train" key contains a series of augmentations
    including resizing, random rotations, flipping, downscaling, shifting, scaling, 
    rotating, hue-saturation-value adjustments, brightness-contrast adjustments, 
    normalization, and tensor conversion. The "valid" key contains resizing, 
    normalization, and tensor conversion.
    """

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

class DataSplit:
    def __init__(self, config:DataTransformationConfig):
        """
        Initialize the DataSplit object with the given configuration.

        Parameters:
        config (DataTransformationConfig): The configuration object containing
        parameters for data transformation including root directory, image folder
        name, file name, test data size, train-validation split, image size, and
        seed value.

        This constructor also calls the _split and _create_datasets methods to
        partition the data and create datasets.
        """

        self.config = config
        self._split()
        self._create_datasets()
       
        
    def _split(self):
        """
        Split the data into train, valid, and test sets.

        This method reads the CSV file located at self.config.root_dir/self.config.file_name
        and splits it into three datasets based on the test_data_size and train_valid_split
        parameters in the configuration. The split datasets are then stored as instance
        variables, which are later used to create the datasets.
        """
        df = pd.read_csv(self.config.root_dir/self.config.file_name)
        df_target = df["target"].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            df, df_target, test_size=self.config.test_data_size, random_state=int(self.config.seed))

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=self.config.train_valid_split, random_state=int(self.config.seed))
        
        self.train_data = X_train, y_train
        self.valid_data = X_valid, y_valid
        self.test_data = X_test, y_test

    def _create_datasets(self):
        """
        Create datasets for training, validation, and test data.

        This method creates instance variables which are datasets for the training,
        validation, and test data. The datasets are created with the KidneyDataset
        class, which is a custom dataset class that loads the image data from the
        file names in the CSV file.
        """
        self.train_dataset = KidneyDataset(self.train_data[0], self.train_data[1], self.config.root_dir, transform)
        self.valid_dataset = KidneyDataset(self.valid_data[0], self.valid_data[1], self.config.root_dir, transform)
        self.test_dataset = KidneyDataset(self.test_data[0], self.test_data[1], self.config.root_dir, transform)

    def get_train_dataset(self):
        """
        Return the training dataset.

        Returns:
        KidneyDataset: The training dataset.
        """
        
        return self.train_dataset
    
    def get_valid_dataset(self):
        """
        Return the validation dataset.

        Returns:
        KidneyDataset: The validation dataset.
        """
        return self.valid_dataset
    
    def get_test_dataset(self):
        """
        Return the test dataset.

        Returns:
        KidneyDataset: The test dataset.
        """
        return self.test_dataset


class KidneyDataset(Dataset):
    def __init__(self, df, y, root_dir, transforms=None):
        """
        Initialize the KidneyDataset class.

        Parameters:
        df (pd.DataFrame): Dataframe containing the image file names and labels.
        y (pd.Series): Series containing the labels.
        root_dir (Path): Path to the root directory containing the images.
        transforms (A.Compose): Albumentations compose object for transformations (default: None).

        Sets the instance variables: df, file_names, targets, transforms, and root_dir.
        """
        self.df = df
        self.file_names = df['path'].values  
        self.targets = y.values
        self.transforms = transforms
        self.root_dir = root_dir
        
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        int: The number of samples in the dataset.
        """

        return len(self.df)
    
    def __getitem__(self, index):
        """
        Return a sample from the dataset.

        Parameters:
        index (int): Index of the sample to return.

        Returns:
        dict: A dictionary containing the image and its corresponding label.
        """
        img_path = self.file_names[index]
        img_path = self.root_dir/img_path
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
    


        



    
        
        