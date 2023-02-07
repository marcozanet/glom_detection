import os
from skimage import io, transform, color
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import utils_image
from bags_creation import BagCreator


class MILDataset(Dataset):
    """ Dataset for bags of instances. 
        train_data = [bag_features, bag_label]
        -> given an idx it reads bag_features, bag_label"""


    def __init__(self, 
                instances_folder:str, 
                exp_folder:str,
                sclerosed_idx: int,
                ) -> None:
        
        
        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."

        self.instances_folders = instances_folder
        self.sclerosed_idx = sclerosed_idx
        self.exp_folder = exp_folder     

        creator = BagCreator(instances_folder=instances_folder, 
                            sclerosed_idx=sclerosed_idx, 
                            exp_folder=exp_folder)
        self.bags_instances, self.bags_features, self.bags_labels = creator()   

        return 


    def __len__(self) -> int:
        return len(self.bags_instances.keys())


    def __getitem__(self, idx) -> dict:
        
        bag_features: np.ndarray
        bag_label: int
        bag_features = np.load(file=self.bags_features[idx])
        bag_label = self.bags_labels[idx]

        print(f"'bag_features', shape: {bag_features.shape}. Bag label: {bag_label}")

        return bag_features, bag_label
    
