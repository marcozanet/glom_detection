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
                n_images_per_bag:int,
                n_classes:int,
                batch = 2, 
                num_workers = 8, 
                # resize = False, 
                mapping: dict = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}) -> Tuple:

    assert os.path.isdir(train_img_dir), f"'train_img_dir':{train_img_dir} is not a validi dirpath."
    # assert os.path.isdir(val_img_dir), f"'val_img_dir':{val_img_dir} is not a validi dirpath."
    assert os.path.isdir(test_img_dir), f"'test_img_dir':{test_img_dir} is not a validi dirpath."
    assert isinstance(batch, int), f"'batch':{batch} should be int."
    assert isinstance(num_workers, int), f"'num_workers':{num_workers} should be int."
    # assert isinstance(resize, bool), f"'resize':{resize} should be boolean."
    assert isinstance(mapping, dict), f"'mapping':{mapping} should be dict."

    # get train, val, test set:
    trainset = MILDataset(instances_folder=train_img_dir, 
                        exp_folder = train_detect_dir,
                        sclerosed_idx=sclerosed_idx,
                        n_images_per_bag = n_images_per_bag,
                        n_classes = n_classes)
    
    print(f"trainset DONEEEE")
    train_dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)

    print(f"trainloader DONEEEE")
    # next(iter(train_dataloader))

    # valset = MILDataset(instances_folder=val_img_dir, 
    #                       exp_folder = val_detect_dir,
    #                       sclerosed_idx=sclerosed_idx)
    valset = MILDataset(instances_folder=val_img_dir, 
                        exp_folder = val_detect_dir,
                        sclerosed_idx=sclerosed_idx,
                        n_images_per_bag = n_images_per_bag,
                        n_classes = n_classes)
    
    # print(f"Train size: {len(trainset)} bags.")
    # # print(f"Valid size: {len(valset)} images." )
    # print(f"Test size: {len(testset)} bags.")

    # print(f"Getting Loaders:")
    # valid_dataloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    # bag_features, bag_label = next(iter(train_dataloader))
    # # print(f"Trainloader:'bag_features':{bag_features.items()}. Bag label: {bag_label}")
    # # bag_features, bag_label = next(iter(valid_dataloader))
    # # print(f"Valloader:'bag_features':{bag_features.shape}. Bag label: {bag_label}")
    # bag_features, bag_label = next(iter(test_dataloader))
    # print(f"Testloader:'bag_features':{bag_features[0].shape}. Bag label: {bag_label}")

    return train_dataloader,val_dataloader #valid_dataloader, 
    

def test_get_loaders():

    train_img_dir = '/Users/marco/helical_tests/test_bagcreator/images'
    val_img_dir = train_img_dir
    test_img_dir = train_img_dir
    train_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
    val_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
    test_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
    batch = 1
    sclerosed_idx = 2
    num_workers = 8
    mapping = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}
    n_images_per_bag = 9
    n_classes = 4
        
    trainloader, testloader= get_loaders(train_img_dir=train_img_dir,
                                        train_detect_dir=train_detect_dir, 
                                        # val_img_dir=val_img_dir,
                                        # val_detect_dir=val_detect_dir,
                                        test_img_dir=test_img_dir,
                                        test_detect_dir=test_detect_dir,
                                        sclerosed_idx=sclerosed_idx,
                                        batch=batch,
                                        n_images_per_bag = n_images_per_bag,
                                        n_classes = n_classes,
                                        num_workers=num_workers,
                                        mapping=mapping)
    
    # print(next(iter(trainloader)))

    # for i, data in trainloader:
    #     X = data[0]
    #     print(X)
    #     raise NotImplementedError()

    return


if __name__ == '__main__':
    test_get_loaders()