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
import matplotlib.pyplot as plt
import torchvision


def prepare_data(cnn_root_fold:str, map_classes:dict, batch:int, num_workers:int, 
                 yolo_root:str, yolo_exp_folds:List[str], resize_crops:bool, treat_as_single_class:bool): 
    """ Prepares data for feature extraction: puts all images in the same folder 
        (regardless of trainset, valset, testset) and gets the dataloader to be used by the model."""
    
    assert 'false_positives' in map_classes.keys(), f"'false_positives' missing in 'map_classes'. "
    cnn_root = os.path.join(cnn_root_fold, 'cnn_dataset')

    # Assign true classes back to crops out of yolo:
    print("-"*10)
    print("Labelling crops from YOLO:")
    print("-"*10)
    for exp_fold in yolo_exp_folds:
        labeller = CropLabeller(root_data=yolo_root, exp_data=exp_fold, map_classes=map_classes, resize = False)
        labeller()
        # raise NotImplementedError()
    print("-"*10)

    # Creating Dataset and splitting into train, val, test:
    print("Creating Dataset and splitting into train, val, test:")
    print("-"*10)
    cnn_processor = CNNDataSplitter(src_folds=yolo_exp_folds, map_classes=map_classes, yolo_root=yolo_root, 
                                    dst_root=cnn_root, resize=resize_crops, treat_as_single_class=treat_as_single_class)
    cnn_processor()
    print("-"*10)

    cnn_dataset_fold = os.path.join(cnn_root_fold, 'cnn_dataset')

    return cnn_dataset_fold


def show_data(images:torch.Tensor, pred_lbl:torch.Tensor, 
              gt_lbl:torch.Tensor, map_classes:dict, n_epoch:int = None):
    """ Plots images during training. """
    
    def imshow(inp:torch.Tensor, title=None, n_epoch:int=None):

        inp = inp.permute((1,2,0)).numpy()
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        fig.savefig(f"cnn_training.png")
        plt.close()
        return

    images = images.cpu()
    pred_lbl = pred_lbl.cpu()
    gt_lbl = gt_lbl.cpu()
    out = torchvision.utils.make_grid(images)
    onehot2int = lambda tensor: int(tensor.numpy())
    reversed_map_classes = {v:k for k,v in map_classes.items()}
    pred_titles = [reversed_map_classes[onehot2int(x)] for x in pred_lbl]
    gt_titles = [reversed_map_classes[onehot2int(x)] for x in gt_lbl]
    title = [f"P:{pr}_GT:{gt}" for pr, gt in zip(pred_titles, gt_titles)]

    imshow(out, title=title, n_epoch=n_epoch)
    
    return





def train_model(model, dataloader_cls, dataloaders, device, map_classes,
                criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    model.to(device)
    print(f"Model is on '{next(model.parameters()).device}'")
    criterion = criterion.to(device)

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
                
            # Use half training dataset
            # if i >= len(dataloaders['train']) / 100:
            #     break

            inputs, labels = data
            t_batch = labels.shape[0]
            inputs = inputs.to(device)
            labels =  labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            _, true_classes = torch.max(labels.data, 1, keepdim=True)


            if epoch%1 == 0 and i==0: 
                show_data(images=inputs, pred_lbl=preds, gt_lbl=true_classes, map_classes=map_classes, n_epoch=epoch)

            loss = criterion(outputs, labels.data)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data
            acc_train += (torch.sum(preds == true_classes).cpu().numpy() / t_batch) # number true classes for all images in batch

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss = loss_train / i #dataloader_cls.trainset_size
        avg_acc = acc_train / i #dataloader_cls.trainset_size
        
        model.train(False)
        model.eval()
            
        for j, data in enumerate(dataloaders['val']):
                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            v_batch = labels.shape[0]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            loss = criterion(outputs, labels.data)
            
            loss_val += loss.data
            acc_val += (torch.sum(preds == labels.data) / v_batch)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / j #dataloader_cls.valset_size
        avg_acc_val = acc_val / j #dataloader_cls.valset_size

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
    
    for k, data in enumerate(tqdm(dataloaders['val'])):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data
        v_batch = labels.shape[0]

        # inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1, keepdim=True)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += (torch.sum(preds == labels.data) / v_batch)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    # print(dataloader_cls.valset_size)
    avg_loss = loss_test / k #dataloader_cls.valset_size
    avg_acc = acc_test / k #dataloader_cls.valset_size
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

    return