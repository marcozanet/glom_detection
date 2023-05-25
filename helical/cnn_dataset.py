import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Literal, List, Tuple
from glob import glob
import os, shutil
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from skimage import io
import matplotlib.pyplot as plt 

class CNNDataset(Dataset): 

    def __init__(self, 
                 root_dir:str,
                 dataset:Literal['train', 'val', 'test'],
                 map_classes:dict,
                 ) -> None:
        
        self.root_dir = root_dir 
        self.dataset = dataset 
        self.map_classes = map_classes
        self.image_list = self._get_image_list()
        self.imgs_fn = [os.path.basename(fp) for fp in self.image_list] if self.image_list is not None else None
        self.transform = self._get_transf_()

        return 
    
    def _get_transf_(self): 
        """ Gets transform depending on the data used . """

        if self.dataset == 'train': 
            transform = A.Compose([
                A.Blur(), 
                A.CLAHE(),
                A.ChannelShuffle(),
                A.RandomBrightnessContrast(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                ToTensorV2(),
                # ConvertImageDtype()
            ])

        return transform
    

    def _get_image_list(self) -> List[str]:
        """ Retrieves all images (image paths) from root folder. """

        assert self.dataset in ['train', 'val', 'test'], f"dataset:{dataset} should be in ['train', 'val', 'test']. "
        image_list = glob(os.path.join(self.root_dir, self.dataset, '*', '*.jpg'))
        image_list = [file for file in image_list if "DS" not in file]

        if len(image_list) == 0: 
            if self.dataset == 'test':
                print(f"Image list for '{self.dataset}' is empty.")
                return None
            else:
                raise ValueError(f"Image list for '{self.dataset}' is empty.")

        return image_list
    
    
    def __len__(self):
        return len(self.image_list) if self.image_list is not None else 0
    
    
    def __getitem__(self, idx) -> None:

        if self.image_list is None: 
            print(f'empty for {self.dataset}')
            return None, None
        
        # image:
        img_fp = self.image_list[idx] 
        img = io.imread(img_fp)
        image = self.transform(image=img)['image']
        image = image.float()
        image = image / 255
        # print(f"max: {image.max()}, min: {image.min()}")

        # label:
        label_name = os.path.split(os.path.dirname(img_fp))[1]
        label_val = self.map_classes[label_name]
        label = torch.tensor(label_val)
        label = nn.functional.one_hot(label, num_classes = len(self.map_classes))
        label = label.float()
        # TODO TRANSFORM THIS IN TORCH

        # print(f"image size: {image.shape}, label size: {label.shape}")
        assert isinstance(image, torch.Tensor), f"{type(image)}"
        assert isinstance(label, torch.Tensor), f"{type(label)}"


        return image, label
    


    


if __name__ == '__main__':
    root_dir = '/Users/marco/helical_tests/test_cnn_processor'
    dataset = 'train'
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0, 'false_positives':2} 
    dataset = CNNDataset(root_dir=root_dir, dataset=dataset, map_classes=map_classes)
    image, _ = dataset[2]
    print(type(image))
    image = image.numpy()
    print(type(image))
    print(image.shape)

    # plt.imshow(image)