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
        bag_features = self.bags_features[idx]
        # print(type(bag_features))
        # print(bag_features)
        
        features = np.load(bag_features[0])
        features = np.expand_dims(features, axis= 0)
        # print(type(features))
        # print(features.shape)

        for i in range(1, len(bag_features)):
            new_feats = np.load(bag_features[i]) # load np file
            new_feats = np.expand_dims(new_feats, axis= 0)
            features = np.concatenate([features, new_feats], axis = 0) # accumulate instance feats

        bag_features = features
        bag_label = self.bags_labels[idx]

        # print(f"'bag_features':{len(bag_features)}, bag_instances:{len(self.bags_instances[idx])}. Bag label: {bag_label}")

        return bag_features, bag_label


def test_MILDataset():
    import random
    instances_folder = '/Users/marco/Downloads/test_folders/test_bagcreator/images'
    exp_folder = '/Users/marco/yolov5/runs/detect/exp7'
    sclerosed_idx=2
    dataset = MILDataset(instances_folder=instances_folder,
                         exp_folder=exp_folder, 
                         sclerosed_idx=sclerosed_idx)
    
    print(f"Dataset has {len(dataset)} bags.")
    num = random.randint(0,len(dataset)-1)
    print(f"trying to get num {num}")
    dataset[num]


    return
    


if __name__ == '__main__':
    test_MILDataset()