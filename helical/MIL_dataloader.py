import torch
from torch.utils.data import DataLoader
import os 
from typing import Tuple
from MIL_dataset import MILDataset
from tqdm import tqdm
from utils import get_config_params


def get_loaders(root:str,
                all_slides_dir:str,
                map_classes:dict, 
                bag_classes:dict, 
                batch:int, 
                n_instances_per_bag:int,
                stain:str = 'pas',
                num_workers:int = 0,
                limit_n_bags_to:int = None)-> Tuple:


    # get train, val, test set:

    
    trainset = MILDataset(folder=os.path.join(root, 'train'), 
                        map_classes=map_classes,
                        bag_classes=bag_classes,
                        all_slides_dir=all_slides_dir,
                        stain=stain, 
                        n_instances_per_bag=n_instances_per_bag,
                        limit_n_bags_to=limit_n_bags_to)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Train size: {len(trainset)} bags.")


    testset = MILDataset(folder=os.path.join(root, 'test'), 
                        map_classes=map_classes,
                        bag_classes=bag_classes,
                        all_slides_dir=all_slides_dir,
                        stain=stain, 
                        n_instances_per_bag=n_instances_per_bag,
                        limit_n_bags_to=limit_n_bags_to)
    testloader = DataLoader(testset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"Test size: {len(testset)} bags.")

    # bag_features, bag_label = next(iter(train_dataloader))
    # # print(f"Trainloader:'bag_features':{bag_features.items()}. Bag label: {bag_label}")
    # # bag_features, bag_label = next(iter(valid_dataloader))
    # # print(f"Valloader:'bag_features':{bag_features.shape}. Bag label: {bag_label}")
    # bag_features, bag_label = next(iter(test_dataloader))
    # print(f"Testloader:'bag_features':{bag_features[0].shape}. Bag label: {bag_label}")

    return trainloader, testloader
    

def test_get_loaders():
    
    params = get_config_params('mil_trainer')

    print('*'*20)
    print("PREPARE DATA FOR MIL TRAINING")
    print('*'*20)
    root = params['root']
    all_slides_dir = params['all_slides_dir']
    map_classes = params['map_classes']
    bag_classes = params['bag_classes']
    bag_classes = {0:0.25, 1:0.5, 2:0.75, 3:1} # TODO ISSUE READING YAML
    n_instances_per_bag = params['n_instances_per_bag']
    stain = params['stain']
    batch = params['batch']
    limit_n_bags_to = ['limit_n_bags_to']
    num_workers = params['num_workers']
    train_loader_path = os.path.join(os.path.dirname(root),  'train_loader.pth')
    val_loader_path = os.path.join(os.path.dirname(root), 'val_loader.pth')
    feat_extract_folder_path = os.path.join(os.path.dirname(root), 'feat_extract')
    # load loaders if they exist:
    if os.path.isdir(feat_extract_folder_path):
        print(f"'feat_extract' folder existing.")
        if os.path.isfile(train_loader_path) and os.path.isfile(val_loader_path):
            print(f"Dataloaders loaded")
            train_loader = torch.load(train_loader_path)
            val_loader = torch.load(val_loader_path)
            # return train_loader, val_loader
    else:
        # if they don't exist already, compute them:
        train_loader, val_loader =  get_loaders(root=root,
                                                all_slides_dir=all_slides_dir,
                                                map_classes=map_classes, 
                                                bag_classes=bag_classes, 
                                                n_instances_per_bag=n_instances_per_bag,
                                                stain=stain,
                                                batch = batch, 
                                                num_workers = num_workers,
                                                limit_n_bags_to=limit_n_bags_to)
        torch.save(train_loader, train_loader_path)
        torch.save(val_loader, val_loader_path)
    
    
    feats, labels = next(iter(train_loader))
    print(f"Bag feats: {feats[0].shape}")
    print(f"Bag label: {labels[0]}")
    feats, labels = next(iter(train_loader))
    print(f"Bag feats: {feats[0].shape}")
    print(f"Bag label: {labels[0]}")
    feats, labels = next(iter(train_loader))
    print(f"Bag feats: {feats[0].shape}")
    print(f"Bag label: {labels[0]}")
    feats, labels = next(iter(train_loader))
    print(f"Bag feats: {feats[0].shape}")
    print(f"Bag label: {labels[0]}")

    print(f"Now printing length")
    print(len(train_loader))


    batches = len(train_loader) 
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)


    return


if __name__ == '__main__':
    test_get_loaders()