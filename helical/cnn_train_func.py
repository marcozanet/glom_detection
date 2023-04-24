from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable

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
        
        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs, labels = data
            inputs.to(device), labels.to(device)
            
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # print(f"ouput data: {outputs.data.shape}")
            # print(outputs.data)
            # print(f"preds: {torch.max(outputs.data, 1, keepdim=True)}")
            
            _, preds = torch.max(outputs.data, 1, keepdim=True)

            # print(f"\noutputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.data.shape}")
            # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
            loss = criterion(outputs, labels.data)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data
            # print(f"labels shape: {labels.data.shape}")

            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataloader_cls.trainset_size
        avg_acc = acc_train * 2 / dataloader_cls.trainset_size
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            # print(f"\noutputs shape: {outputs.shape}")
            # print(f"labels shape: {labels.data.shape}")
            # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
            loss = criterion(outputs, labels.data)
            
            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / dataloader_cls.valset_size
        avg_acc_val = acc_val / dataloader_cls.valset_size
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
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

    test_batches = len(dataloaders['test'])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders['test']):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data

        inputs, labels = data
        inputs.to(device), labels.to(device)

        outputs = model(inputs)
        # print(f"\noutputs shape: {outputs.shape}")
        # print(f"labels shape: {labels.data.shape}")
        # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
        _, preds = torch.max(outputs.data, 1, keepdim=True)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataloader_cls.testset_size
    avg_acc = acc_test / dataloader_cls.testset_size
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)