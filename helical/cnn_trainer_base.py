import torch
import time
import os, shutil
import copy
from typing import List
import math
import numpy as np
from tqdm import tqdm 
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from cnn_loaders import CNNDataLoaders
from utils import get_config_params
from configurator import Configurator


class CNN_Trainer_Base(Configurator):

    def __init__(self, config_yaml_fp:str) -> None:

        super().__init__()
        self.config_yaml_fp = config_yaml_fp
        self.class_name = self.__class__.__name__
        self.params = get_config_params(config_yaml_fp,'cnn_trainer')
        self._set_all_attributes()
        return
    
    
    def _set_all_attributes(self)->None:
        """ Sets all attributes contained in self.params. """
        func_n = self._set_all_attributes.__name__
        self.yolo_data_root = self.params['yolo_data_root']
        self.cnn_data_fold = self.params['cnn_data_fold']
        self.map_classes = self.params['map_classes'] if 'false_positives' in self.params['map_classes'].keys() else self._get_map_classes(self.params['map_classes'])
        self.lr = self.params['lr']
        self.k_tot = self.params['k_tot']
        self.k_i = self.params['k_i']
        self.batch = self.params['batch']
        self.epochs = self.params['epochs']
        self.num_workers = self.params['num_workers']
        self.weights_path = self.params['weights_path']
        self.cnn_exp_fold = self.params['cnn_exp_fold']
        self.dataset = self.params['dataset']
        self.yolo_task = self.params['yolo_task']
        self.crop_percentile = self.params['crop_percentile'] if self.params['crop_percentile'] is not None else 90
        self.treat_as_single_class = self.params['treat_as_single_class']
        self.device = self.params['device']
        self.gpu = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device == 'mps':
            # Check that MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    self.format_msg("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.", func_n=func_n, type='warning')
                else:
                    self.format_msg("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.", func_n=func_n, type='warning')
        self.device = torch.device('mps') if self.device == 'mps' else self.gpu
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%Y_%m_%d__%H_%M_%S")
        self.weights_save_fold = os.path.join(self.cnn_exp_fold,f"{self.dt_string}")
        self.yolov5dir = self.params['yolov5dir']
        self.skip_test = self.params['skip_test']
        self.yolo_exp_folds = self._get_last3_detect_dir()
        self.skipped_crops = []
        self.min_w_h = self.params['min_w_h'] if self.params['min_w_h'] is not None else 0.1
        self.mode= self.params['mode']
        
        return
    

    def _get_map_classes(self, old_map_classes:dict):

        new_map_classes = {k:v for k,v in old_map_classes.items()}
        new_map_classes.update({'false_positives':max(new_map_classes.values())+1}) # adding false positive class for base class = 0 e.g. Glomerulus (wo classification)
        return new_map_classes
    

    def _parse_args(self)->None:
        assert os.path.isdir(self.yolo_data_root), self.assert_log(f"'yolo_data_root':{self.yolo_data_root} is not a valid dirpath.")
        return
    

    def get_loaders(self, mode:str, show_data:bool=True)->None:
        """ Creates DataLoader class and gets Train and Val Loaders. """
        func_n = self.get_loaders.__name__
        assert mode in ['train', 'val', 'inference'], self.assert_log(f"'mode':{mode} should be one of ['train', 'val', 'inference']", func_n=func_n)
        self.format_msg(f"⏳ Getting loaders", func_n=func_n)
        self.dataloader_cls = CNNDataLoaders(root_dir=os.path.join(self.cnn_data_fold, 'cnn_dataset'),
                                             map_classes=self.map_classes, mode=mode,
                                             batch=self.batch, num_workers=self.num_workers)
        if mode=='inference': show_data=False
        self.loaders = self.dataloader_cls(show_data=show_data)
        self.format_msg(f"✅ Got loaders.", func_n=func_n)
        return
        

    def _get_last3_detect_dir(self)-> list:
        """ Returns last 3 (modified) folders from exp fold. """
        detect_dir = os.path.join(self.yolov5dir, 'runs', 'detect')
        list_folds = [os.path.join(detect_dir, subfold) for subfold in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, subfold)) ]
        last_3 = []
        n_folds = 3 if self.skip_test is False else 2
        for _ in range(n_folds):
            last_fold = max(list_folds, key=os.path.getctime)
            last_3.append(last_fold)
            list_folds.remove(last_fold)
        return  last_3
    


    def show_train_data(self, images:torch.Tensor, pred_lbl:torch.Tensor,
                  gt_lbl:torch.Tensor, ncols:int=2, mode:str='train')->None:
        """ Plots images during training. """
        
        shorten_name = lambda name: name.split(' ')[0][:4]
        images = images.cpu()
        images = images.permute((0,2,3,1)).numpy()
        pred_lbl, gt_lbl = pred_lbl.cpu(), gt_lbl.cpu()
        reversed_map_classes = {v:k for k,v in self.map_classes.items()}
        pred_titles = [reversed_map_classes[int(x)] for x in pred_lbl]
        gt_titles = [reversed_map_classes[int(x)] for x in gt_lbl]
        titles = [f"P:{shorten_name(pr)}_GT:{shorten_name(gt)}" for pr, gt in zip(pred_titles, gt_titles)]
        colors = ['green' if pr==gt else 'red' for pr, gt in zip(pred_titles, gt_titles) ]
        n_imgs = len(images)
        nrows= math.ceil(n_imgs/ncols)
        fig = plt.figure(figsize=(ncols*3, nrows*3))
        inp_clss=zip(images, titles, colors)

        for i,(img,clss,color) in enumerate(inp_clss):
            axes = plt.subplot(nrows, ncols, i+1)
            plt.imshow(img)
            plt.title(clss)
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])
            
            axes.tick_params(color=color, labelcolor=color)
            for spine in axes.spines.values():
                spine.set_edgecolor(color)

        # plt.show()
        fig.savefig(f'img_cnn_{mode}_preds.png')
        plt.close()
        return
    
    
    def plot_metrics(self, epoch:int, train_losses:list, val_losses:list, train_accs:list, val_accs:list):
        
        if epoch==0: return 
        x = range(epoch+1)
        metrics = [(train_accs, val_accs, 'Accuracy'), (train_losses, val_losses, 'Loss')]
        fig = plt.figure()
        for i, (train_metric, val_metric, name) in enumerate(metrics):
            plt.subplot(2,1,i+1)
            plt.plot(x, train_metric, 'orange')
            plt.plot(x, val_metric, 'green')
            if i==0: plt.yticks(np.arange(0,1.05,0.1))
            plt.xticks(range(self.params['epochs']))
        plt.suptitle(f"CNN Metrics Epoch:{epoch}")
        fig.savefig(f"img_cnn_metrics.png")
        plt.close()
        return
    

    def get_model(self)->None:
        """ Creates VGG16 model with default weights and matches 
            classification head with the desired number of classes. """
        func_n = self.get_model.__name__
        # Load the pretrained model from pytorch
        print("-"*10)
        self.format_msg(f"⏳ Creating VGG16 model.", func_n=func_n)
        vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
        vgg16.load_state_dict(torch.load(self.weights_path))
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[-1]= nn.Linear(num_features, len(self.map_classes))
        self.format_msg(f"✅ Created VGG16 model.", func_n=func_n)
        return vgg16
    
    

    def train_model(self)->torch.Tensor:
        """ Trains the VGG model. """
        func_n = self.train_model.__name__

        # set starting values
        model = self.get_model()
        assert any([param.requires_grad for param in model.features.parameters()]), self.assert_log(f"No param requires grad. Model won't update.", func_n=func_n)
        if self.device != 'cpu': model.to(self.device)   # move model to gpus
        criterion = nn.CrossEntropyLoss()
        since = time.time() # get start time
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        self.log.info(f"lr: {self.lr}")
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.format_msg(msg=f"Model is on: {next(model.parameters()).device}", func_n=func_n)
        # self.format_msg(msg=f"'criterion' is on: {criterion.device}", func_n=func_n)

        # TRAINING LOOP
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        get_desc = lambda acc_train_i, loss_train_i: f"Acc train: {acc_train_i:.2f}, Loss train: {loss_train_i:.2f}."
        progress_bar = tqdm(self.loaders['train'], desc=get_desc(acc_train_i=0, loss_train_i=0))
        for epoch in range(self.epochs):

            # 1) TRAIN for each epoch:
            self.format_msg(msg=f"⏳ Epoch {epoch}/{self.epochs}", func_n=func_n)
            print("⏳ Epoch {}/{}".format(epoch, self.epochs))
            print('-' * 10)
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_train_i = 0
            acc_val = 0
            model.train(True)
            # for each batch in train loaders:
            for i, data in enumerate(progress_bar):
                inputs, labels = data
                t_batch = labels.shape[0]
                if self.device != 'cpu': inputs = inputs.to(self.device)     # move images to gpu
                if self.device != 'cpu': labels = labels.to(self.device)    # move labels to gpu
                optimizer.zero_grad()               # zero out gradient
                # model.zero_grad()
                outputs = model(inputs)             # run input through model and get output
                _, preds = torch.max(outputs.data, 1, keepdim=True)         # get pred from logits
                _, true_classes = torch.max(labels.data, 1, keepdim=True)   # get true class 
                # show data
                if i==0:  self.show_train_data(images=inputs, pred_lbl=preds, gt_lbl=true_classes, n_epoch=epoch)
                weights_before = list(model.parameters())[0].clone()
                loss = criterion(outputs, labels)     # evaluate model
                loss.backward()                             # update loss (graph)
                optimizer.step()                           # update optimizer
                weights_after = list(model.parameters())[0].clone()
                # check that weights are being updated
                # if i==2: assert not {torch.equal(weights_before.data, weights_after.data)}, self.assert_log(f"Weights are not being updated.", func_n=func_n)
                loss_train += loss.data                    # update mean loss 
                acc_train_i = torch.sum(preds == true_classes).cpu().numpy()/t_batch
                acc_train += acc_train_i 
                if i%1==0: progress_bar.set_description(desc=get_desc(acc_train_i=acc_train_i, loss_train_i=loss.data))
                if acc_train>len(self.loaders['train']): print(f"acc_train_i:{acc_train:.4f}. torch_sum:{torch.sum(preds == true_classes).cpu().numpy()}/{t_batch}")
                del inputs, labels, outputs, preds # free cache memory
                torch.cuda.empty_cache()
            avg_loss = loss_train / (i+1)   # average loss over epochs 
            avg_acc = acc_train / (i+1)     # average acc over epochs
            train_losses.append(avg_loss.cpu())
            train_accs.append(avg_acc)

            # 2) VALIDATE for each epoch:
            model.train(False)
            model.eval()
            for j, data in enumerate(self.loaders['val']):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device) # move to gpu
                v_batch = labels.shape[0]
                optimizer.zero_grad()       # zero out gradient
                outputs = model(inputs)     # run input through model and get output
                _, preds = torch.max(outputs.data, 1, keepdim=True)
                _, true_classes = torch.max(labels.data, 1, keepdim=True)
                loss = criterion(outputs, labels.data)
                loss_val += loss.data
                acc_val += (torch.sum(preds == true_classes).cpu().numpy() / v_batch)
                if acc_val>len(self.loaders['val']): print(f"acc_val_i:{acc_val:.4f}. torch_sum:{torch.sum(preds == true_classes).cpu().numpy()}/{v_batch}")
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            avg_loss_val = loss_val / (j+1) 
            avg_acc_val = acc_val / (j+1) 
            val_losses.append(avg_loss_val.cpu())
            val_accs.append(avg_acc_val)

            # 3) POST EPOCH SAVING:
            self.format_msg(msg=f"✅ Epoch {epoch}/{self.epochs}. Metrics:", func_n=func_n)
            self.format_msg(msg=f"Avg loss (train): {avg_loss:.4f}, Avg loss (val): {avg_loss_val:.4f}", func_n=func_n)
            self.format_msg(msg=f"Avg acc (train): {avg_acc:.4f}, Avg acc (val): {avg_acc_val:.4f} ", func_n=func_n)
            print(f"Epoch {epoch} result: ")
            print(f"Avg loss (train): {avg_loss:.4f}, Avg loss (val): {avg_loss_val:.4f} ")
            print(f"Avg acc (train): {avg_acc:.4f}, Avg acc (val): {avg_acc_val:.4f} ")
            print('-' * 10)
            # plot metrics:
            self.plot_metrics(epoch=epoch, train_accs=train_accs, train_losses=train_losses, 
                              val_accs=val_accs, val_losses=val_losses)
            # update best model:
            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                best_model_wts = copy.deepcopy(model.state_dict())
            

        # print trainining results:
        elapsed_time = time.time() - since
        msg = "Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60)
        self.format_msg(msg=msg, func_n=func_n)
        msg = "Best acc: {:.4f}".format(best_acc)
        self.format_msg(msg=msg, func_n=func_n)
        model.load_state_dict(best_model_wts)   # save best model
        return model

