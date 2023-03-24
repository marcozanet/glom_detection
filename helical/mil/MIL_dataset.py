import os
from skimage import io, transform, color
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from bags_creation import BagCreator
from configurator import Configurator
import random
from torch.utils.data import DataLoader


class MILDataset(Dataset, Configurator):
    """ Dataset for bags of instances. 
        train_data = [bag_features, bag_label]
        -> given an idx it reads bag_features, bag_label"""


    def __init__(self, 
                instances_folder:str, 
                exp_folder:str,
                sclerosed_idx: int,
                n_images_per_bag: int,
                n_classes: int,
                ) -> None:
        
        
        self.log = self.get_logger()

        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."

        self.instances_folders = instances_folder
        self.sclerosed_idx = sclerosed_idx
        self.exp_folder = exp_folder
        self.n_images_per_bag = n_images_per_bag
        self.n_classes = n_classes
        self.class_name = self.__class__.__name__ 

        # creator = BagCreator(instances_folder=instances_folder, 
        #                     sclerosed_idx=sclerosed_idx, 
        #                     exp_folder=exp_folder,
        #                     n_images_per_bag=n_images_per_bag,
        #                     n_classes=n_classes)
        # self.bags_indices, self.bags_features, self.bags_labels = creator()
        # self.log_example()
        self.data = self.get_data()
        return 
    
    
    def get_data(self):

        creator = BagCreator(instances_folder=self.instances_folders, 
                            sclerosed_idx=self.sclerosed_idx, 
                            exp_folder=self.exp_folder,
                            n_images_per_bag=self.n_images_per_bag,
                            n_classes=self.n_classes)
        bags_indices, bags_features, bags_labels = creator()
        data = []
        for bag, images in bags_indices.items(): 
            # print(bag)
            bag_features = bags_features[bag]
            # print(bag_features)
            bag_label = bags_labels[bag]
            # print(bag_label)
            # print(bag_features.keys())
            data = [(bag_features[i], bag_label) for i in bag_features.keys()]

        return  data
    
    # def log_example(self):
    #     """ Logs example of output from dataset."""
        
    #     ex_bag_features, ex_bag_label = self.__getitem__(0)
    #     rand_idx_image = random.choice(list(ex_bag_features.keys()))
    #     self.log.info(f"{self.class_name}.{'__init__'}: example output of dataset[idx_bag=0]: bag_features: {len(ex_bag_features.keys())} features, each of shape:{ex_bag_features[rand_idx_image].shape}, bag_label:{ex_bag_label} ")
        


    #     return


    def __len__(self) -> int:
        # print(len(self.bags_indices.keys()))
        return len(self.data)


    def __getitem__(self, idx) -> dict:

        print(f"idx: {idx}")
        bag_feat_file = self.data[idx][0]
        # print(bag_feat_file)
        bag_feature = torch.from_numpy(np.load(bag_feat_file))

        # brutally averaging!!
        self.log.warning(f"{self.class_name}.{'__getitem__'}: before: {bag_feature.shape}")
        bag_feature = bag_feature.mean(dim=0)
        self.log.warning(f"{self.class_name}.{'__getitem__'}: after: {bag_feature.shape}")

        print(f"bag_feat_shape: {bag_feature.shape}")
        # print(bag_feature)
        bag_label = torch.Tensor(self.data[idx][1])



        self.log.warning(f"{self.class_name}.{'__getitem__'}: brutally averaging to lower dims!!")
        self.log.warning(f"{self.class_name}.{'__getitem__'}: add normalization!!")

        # bag_features = {i: torch.from_numpy(np.load(file)) for i,file in enumerate(bag_feats_files.values())}
        # bag_features = [torch.from_numpy(np.load(file)) for file in bag_feats_files.values()]
        # bag_features = [torch.from_numpy(np.load(bag_feats_files[i])) for i in range(0, len(bag_feats_files))]
        # bag_features = torch.stack(bag_features)
        # bag_features = {i: torch.from_numpy(np.load(bag_feats_files[i])) for i in range(0, len(bag_feats_files)) }
        # print(self.bags_labels)
        # print(self.bags_labels[idx])
        # bag_label = torch.tensor(self.bags_labels[idx])


        return bag_feature, bag_label


def test_MILDataset():

    import random
    instances_folder = '/Users/marco/helical_tests/test_bagcreator/images'
    exp_folder = '/Users/marco/yolov5/runs/detect/exp7'
    sclerosed_idx=2
    n_images_per_bag = 9
    n_classes = 4


    dataset = MILDataset(instances_folder=instances_folder,
                         exp_folder=exp_folder, 
                         sclerosed_idx=sclerosed_idx,
                         n_images_per_bag = n_images_per_bag,
                         n_classes=n_classes)
    print(dataset[0])
    # dict
    # for i in range(len(dataset)):
    #     bag_feats, bag_label = dataset[i]
    #     idx_image = random.choice(list(bag_feats.keys()))
    #     print(f"Feats shape: {bag_feats[idx_image].shape}")
    #     print(f"Label: {bag_label}")

    # list
    # for i in range(len(dataset)):
    #     bag_feats, bag_label = dataset[i]
    #     print(f"Feats shape: {bag_feats[0].shape}")
    #     print(f"Label: {bag_label}")



    return
    


if __name__ == '__main__':
    test_MILDataset()