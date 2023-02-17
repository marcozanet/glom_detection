import torch
from torch.utils.data import DataLoader
import os 
from typing import Tuple
from MIL_dataset import MILDataset


def get_loaders(train_img_dir, 
                train_detect_dir,
                val_img_dir, 
                val_detect_dir,
                test_img_dir,
                test_detect_dir,
                sclerosed_idx:int,
                batch = 2, 
                num_workers = 8, 
                # resize = False, 
                classes = 1,
                mapping: dict = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}) -> Tuple:

    assert os.path.isdir(train_img_dir), f"'train_img_dir':{train_img_dir} is not a validi dirpath."
    assert os.path.isdir(val_img_dir), f"'val_img_dir':{val_img_dir} is not a validi dirpath."
    assert os.path.isdir(test_img_dir), f"'test_img_dir':{test_img_dir} is not a validi dirpath."
    assert isinstance(batch, int), f"'batch':{batch} should be int."
    assert isinstance(num_workers, int), f"'num_workers':{num_workers} should be int."
    # assert isinstance(resize, bool), f"'resize':{resize} should be boolean."
    assert isinstance(classes, int), f"'classes':{classes} should be int."
    assert isinstance(mapping, dict), f"'mapping':{mapping} should be dict."

    # get train, val, test set:
    trainset = MILDataset(instances_folder=train_img_dir, 
                          exp_folder = train_img_dir,
                          sclerosed_idx=sclerosed_idx)
    valset = MILDataset(instances_folder=val_img_dir, 
                          exp_folder = val_detect_dir,
                          sclerosed_idx=sclerosed_idx)
    testset = MILDataset(instances_folder=test_img_dir, 
                          exp_folder = test_detect_dir,
                          sclerosed_idx=sclerosed_idx)

    print(f"Train size: {len(trainset)} images.")
    print(f"Valid size: {len(valset)} images." )
    print(f"Test size: {len(testset)} images.")

    train_dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    data = next(iter(train_dataloader))
    image = data['image']
    print(f"image shape: {image.shape}")
    

    return train_dataloader, valid_dataloader, test_dataloader

def test_get_loaders():

    train_img_dir = '/Users/marco/Downloads/test_folders/test_bagcreator/images'
    val_img_dir = train_img_dir
    test_img_dir = train_img_dir
    batch = 2
    num_workers = 8
    classes = 1
    mapping = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}
        
    get_loaders(train_img_dir=train_img_dir, 
                val_img_dir=val_img_dir,
                test_img_dir=test_img_dir,
                batch=batch,
                num_workers=num_workers,
                classes=classes,
                mapping=mapping)

    return


if __name__ == '__main__':
    test_get_loaders()