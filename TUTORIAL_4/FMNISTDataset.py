from flautim.pytorch.Dataset import Dataset
import numpy as np
import pandas as pd
import torch
import copy
from torch.utils.data import DataLoader
    
class FMNISTDataset(Dataset):

    def __init__(self, FM_Normalization, EVAL_Transforms, TRAIN_Transforms, partition, **kwargs):
    
        name = kwargs.get('name', 'FMNIST')
    
        super(FMNISTDataset, self).__init__(name, **kwargs)
        
        self.FM_Normalization = FM_Normalization
        self.EVAL_Transforms = EVAL_Transforms
        self.TRAIN_Transforms = TRAIN_Transforms
        
        self.feature_name = kwargs.get("feature_name", 'image')
        self.split = kwargs.get("split_data", True)
        
        if self.split:
            partition = partition.train_test_split(test_size=0.2, seed=42)

        self.train_partition = partition["train"]
        self.test_partition = partition["test"]
        
    def dataloader(self, validation = False):
        tmp = self.validation() if validation else self.train()
        return DataLoader(tmp, batch_size = self.batch_size, num_workers = 1)
    
        
    
    def apply_train_transforms(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[self.feature_name] = [self.TRAIN_Transforms(img) for img in batch[self.feature_name]]
        return batch


    def  apply_eval_transforms(self, batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[self.feature_name] = [self.EVAL_Transforms(img) for img in batch[self.feature_name]]
        return batch


    def train(self):

        return self.train_partition.with_transform(self.apply_train_transforms)


    def validation(self):
        
        return self.test_partition.with_transform(self.apply_eval_transforms)
 