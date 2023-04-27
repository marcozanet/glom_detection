import torch
import time
import os
import copy
from typing import List
from cnn_assign_class_crop import CropLabeller
from cnn_splitter import CNNDataSplitter
from cnn_loaders import CNNDataLoaders
from glob import glob
from tqdm import tqdm 
import shutil


def prepare_data(cnn_root_fold:str, map_classes:dict, batch:int, num_workers:int, 
                 yolo_root:str, yolo_exp_folds:List[str], resize_crops:bool): 
    """ Prepares data for feature extraction: puts all images in the same folder 
        (regardless of trainset, valset, testset) and gets the dataloader to be used by the model."""
    
    assert 'false_positives' in map_classes.keys(), f"'false_positives' missing in 'map_classes'. "
    cnn_root = os.path.join(cnn_root_fold, 'cnn_dataset')

    # Assign true classes back to crops out of yolo:
    print("Labelling crops from YOLO:")
    print("-"*10)
    for exp_fold in yolo_exp_folds:
        labeller = CropLabeller(root_data=yolo_root, exp_data=exp_fold, map_classes=map_classes, resize = False)
        labeller()

    # Creating Dataset and splitting into train, val, test:
    print("Creating Dataset and splitting into train, val, test:")
    print("-"*10)
    cnn_processor = CNNDataSplitter(src_folds=yolo_exp_folds, map_classes=map_classes, yolo_root=yolo_root, 
                                    dst_root=cnn_root, resize=resize_crops)
    cnn_processor()

    cnn_dataset_fold = os.path.join(cnn_root_fold, 'cnn_dataset')

    return cnn_dataset_fold

    # # Putting images all in same fold and extracting features:
    # print("Create Dataset for Feature Extraction:")
    # print("-"*10)
    # feat_extract_fold = os.path.join(cnn_root_fold, 'feat_extract')
    # os.makedirs(feat_extract_fold, exist_ok=True)
    # # get all images: 
    # images = glob(os.path.join(cnn_root, '*', '*', '*.jpg')) # get all images 
    # class_fold_names = glob(os.path.join(cnn_root, '*', '*/'))
    # class_fold_names = set([os.path.split(os.path.dirname(fp))[1] for fp in class_fold_names])
    # # print(class_fold_names)

    # assert len(images)>0, f"No images like {os.path.join(cnn_root, '*', '*', '*.jpg')}"
    # # create folds:
    # for fold in class_fold_names: 
    #     os.makedirs(os.path.join(feat_extract_fold, fold), exist_ok=True)
    # # fill fold:
    # for img in tqdm(images, desc="Filling 'feature_extract'"): 
    #     clss_fold_name = os.path.split(os.path.dirname(img))[1]
    #     dst = os.path.join(feat_extract_fold, clss_fold_name, os.path.basename(img))
    #     if not os.path.isfile(dst):
    #         shutil.copy(src=img, dst=dst)
    # assert len(os.listdir(feat_extract_fold))>0, f"No images found in extract fold: {feat_extract_fold}"
    # dataloader_cls = CNNDataLoaders(root_dir=feat_extract_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
    # dataloader = dataloader_cls()

    # return  dataloader


def train_model(model, dataloader_cls, dataloaders, 
                criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, data in enumerate(tqdm(dataloaders['train'])):
            # if i % 100 == 0:
            #     print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # # Use half training dataset
            # if i >= train_batches / 2:
            #     break
            inputs, labels = data
            t_batch = labels.shape[0]
            inputs.to(device), labels.to(device)
            
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # print(f"ouput data: {outputs.data.shape}")
            # print(outputs.data)
            # print(f"preds: {torch.max(outputs.data, 1, keepdim=True)}")
            
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            _, true_classes = torch.max(labels.data, 1, keepdim=True)

            # print(f"\nlabels:{labels}")
            # print(f"\nouputs:{outputs.data}")
            # print(f"\npreds:{preds}")
            # print(f"\ntrue classes:{true_classes}")

            # raise NotImplementedError()
            # print(f"\noutputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.data.shape}")
            # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
            loss = criterion(outputs, labels.data)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data
            # print(f"labels shape: {labels.data.shape}")
            # print(torch.sum(preds == true_classes))
            acc_train += (torch.sum(preds == true_classes) / t_batch) # number true classes for all images in batch

            # raise NotImplementedError()
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss = loss_train / dataloader_cls.trainset_size
        avg_acc = acc_train / dataloader_cls.trainset_size
        # print(f"train acc: {acc_train} / {dataloader_cls.trainset_size} = {avg_acc} ")
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(dataloaders['val']):
            # if i % 100 == 0:
            #     print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            inputs.to(device), labels.to(device)
            v_batch = labels.shape[0]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            # print(f"\noutputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.data.shape}")
            # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
            loss = criterion(outputs, labels.data)
            
            loss_val += loss.data
            acc_val += (torch.sum(preds == labels.data) / v_batch)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / dataloader_cls.valset_size
        avg_acc_val = acc_val / dataloader_cls.valset_size
        # print(f"val acc: {acc_val} / {dataloader_cls.valset_size} = {avg_acc_val} ")

        print(f"Epoch {epoch} result: ")
        print(f"Avg loss (train): {avg_loss:.4f}, Avg loss (val): {avg_loss_val:.4f} ")
        print(f"Avg acc (train): {avg_acc:.4f}, Avg acc (val): {avg_acc_val:.4f} ")
        print('-' * 10)
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model


def eval_model(model, dataloader_cls, dataloaders, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    test_batches = len(dataloaders['val'])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(tqdm(dataloaders['val'])):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data
        v_batch = labels.shape[0]

        # inputs, labels = data
        inputs.to(device), labels.to(device)

        outputs = model(inputs)
        # print(f"\noutputs shape: {outputs.shape}")
        # print(f"labels shape: {labels.data.shape}")
        # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
        _, preds = torch.max(outputs.data, 1, keepdim=True)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += (torch.sum(preds == labels.data) / v_batch)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    print(dataloader_cls.valset_size)
    avg_loss = loss_test / dataloader_cls.valset_size
    avg_acc = acc_test / dataloader_cls.valset_size
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)