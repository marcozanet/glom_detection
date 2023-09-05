import torch
import time
import os, shutil
import copy
from typing import List
from cnn_assign_class_crop_new import CropLabeller
from cnn_splitter import CNNDataSplitter
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torchvision
from loggers import get_logger
from utils import get_config_params
from datetime import datetime
from configurator import Configurator
import math
import numpy as np
from glob import glob
import cv2
from torch import nn


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
        self.weights_save_fold = self.cnn_exp_fold +f"_{self.dt_string}"
        self.yolov5dir = self.params['yolov5dir']
        self.skip_test = self.params['skip_test']
        self.yolo_exp_folds = self._get_last3_detect_dir()
        self.skipped_crops = []
        self.min_w_h = self.params['min_w_h'] if self.params['min_w_h'] is not None else 0.1
        
        return
    

    def _get_map_classes(self, old_map_classes:dict):

        new_map_classes = {k:v for k,v in old_map_classes.items()}
        new_map_classes.update({'false_positives':max(new_map_classes.values())+1}) # adding false positive class for base class = 0 e.g. Glomerulus (wo classification)
        return new_map_classes

    

    def _parse_args(self)->None:

        assert os.path.isdir(self.yolo_data_root), self.assert_log(f"'yolo_data_root':{self.yolo_data_root} is not a valid dirpath.")


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
    
    
    def _get_objs_from_row_txt_label(self, row:str): # helper func
        row = row.replace('\n', '')
        nums = row.split(' ')
        clss = int(float(nums[0]))
        nums = [float(num) for num in nums[1:]]
        # detection case:
        if len(nums) == 4:
            x_c, y_c, w, h = nums
        # segmentation case:
        elif len(nums) == 8: 
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = nums
            x_c, y_c = x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2
            w, h = x_max-x_min, y_max-y_min
            assert all([el>=0 for el in [x_c, y_c, w, h]])
            assert x_c-w/2 == x_min, f"{x_c}-{w}/2 != {x_min}. Result is: {x_c-w/2}. "
        else:
            print(f"there should be 4 or 8 objects apart from class but are {len(nums)}")

        return clss, x_c, y_c, w, h    
    
    
    def prepare_data(self)->None:
        """ Prepares data for CNN training: puts all images in the same folder 
            (regardless of trainset, valset, testset) and gets the dataloader 
            to be used by the model."""
        func_n = self.prepare_data.__name__
        msg_base = f"{self.class_name}.{func_n}: "
        self.log.info(msg_base + f"⏳ Preparing data for CNN training:")


        # Assign true classes back to crops out of yolo:
        # assert 'false_positives' in self.map_classes.keys(), f"'false_positives' missing in 'map_classes'. "
        cnn_root = os.path.join(self.cnn_data_fold, 'cnn_dataset')
        self.log.info(msg_base + f"⏳ Labelling crops out of YOLO:")
        for exp_fold in self.yolo_exp_folds:
            # 1) create crops:
            self.center_crop(exp_fold=exp_fold)
            # 2) assign each crop the correct label:
            labeller = CropLabeller(self.config_yaml_fp, exp_fold=exp_fold, skipped_crops=self.skipped_crops)
            labeller()
        self.log.info(msg_base + f"✅ Labelled crops.")

        # Creating Dataset and splitting into train, val, test:
        self.log.info(msg_base + f"⏳ Creating train,val,test sets:")
        cnn_processor = CNNDataSplitter(src_folds=self.yolo_exp_folds, map_classes=self.map_classes, yolo_root=self.yolo_data_root, 
                                        dst_root=cnn_root, treat_as_single_class=self.treat_as_single_class)
        cnn_processor()
        cnn_dataset_fold = os.path.join(self.cnn_data_fold, 'cnn_dataset')
        self.log.info(msg_base + f"✅ Created train,val,test sets.")

        return cnn_dataset_fold
    


    # def crop_gloms(self, exp_fold:str):

    #     func_n = self.crop_gloms.__name__
    #     pred_label_dir = os.path.join(exp_fold, 'labels')
    #     crops_dir = os.path.join(exp_fold, 'crops')
    #     crops_true_classes_dir = os.path.join(exp_fold, 'crops_true_classes')
    #     images = glob(os.path.join(self.yolo_data_root, 'tiles', '*', 'images', '*.png'))
    #     fnames = [os.path.basename(file).split('.')[0] for file in images]
    #     pred_labels = glob(os.path.join(pred_label_dir, '*.txt'))
    #     reversed_map_classes = {v:k for k,v in self.params['map_classes'].items()}

    #     # if crops already exist, remove and replace:
    #     if os.path.isdir(crops_dir): shutil.rmtree(crops_dir)
    #     if os.path.isdir(crops_true_classes_dir): shutil.rmtree(crops_true_classes_dir)
    #     os.makedirs(crops_dir)

    #     # for each pred, look for corresponding image
    #     all_w, all_h = [], []
    #     for pred_lbl in tqdm(pred_labels, desc='Cropping Images'):
    #         lbl_fn = os.path.basename(pred_lbl).split('.')[0]
    #         try:
    #             idx = fnames.index(lbl_fn)
    #         except:
    #             raise Exception(f"Index not found for '{lbl_fn} in {fnames}")
    #         corr_img = images[idx]
    #         image = cv2.imread(corr_img)
    #         W,H = image.shape[:2]
    #         with open(pred_lbl, 'r') as f:
    #             rows = f.readlines()
            
    #         # for each obj, create a new cropped image:
    #         for i, row in enumerate(rows):
    #             new_image = np.zeros_like(image)
    #             clss, x_c, y_c, w, h = self._get_objs_from_row_txt_label(row=row)
    #             all_w.append(w)
    #             all_h.append(h)
    #             x_min, x_max = int((x_c-w/2)*W), int((x_c+w/2)*W)
    #             y_min, y_max = int((y_c-h/2)*H), int((y_c+h/2)*H)
    #             # saving WITH INVERTED COORDS:
    #             new_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
    #             os.makedirs(os.path.join(crops_dir, reversed_map_classes[clss]), exist_ok=True)
    #             fp = os.path.join(crops_dir, reversed_map_classes[clss], f"{lbl_fn}_crop{i}.jpg")
    #             cv2.imwrite(fp, new_image)

    #     self.all_w, self.all_h = all_w, all_h
    #     self.center_crop(exp_fold=exp_fold)

    #     return

    def get_max_crop(self, exp_fold:str):

        func_n = self.get_max_crop.__name__
        pred_label_dir = os.path.join(exp_fold, 'labels')
        crops_dir = os.path.join(exp_fold, 'crops')
        crops_true_classes_dir = os.path.join(exp_fold, 'crops_true_classes')
        pred_labels = glob(os.path.join(pred_label_dir, '*.txt'))
        # if crops already exist, remove and replace:
        if os.path.isdir(crops_dir): shutil.rmtree(crops_dir)
        if os.path.isdir(crops_true_classes_dir): shutil.rmtree(crops_true_classes_dir)
        os.makedirs(crops_dir)
        # for each pred, look for corresponding image
        all_w, all_h = [], []
        for pred_lbl in tqdm(pred_labels, desc='Cropping Images'):
            with open(pred_lbl, 'r') as f:
                rows = f.readlines()
            # for each obj, create a new cropped image:
            for i, row in enumerate(rows):
                clss, x_c, y_c, w, h = self._get_objs_from_row_txt_label(row=row)
                all_w.append(w)
                all_h.append(h)

        return all_w, all_h
    

    def center_crop(self, exp_fold:str)-> None:

        func_n = self.center_crop.__name__
        all_w, all_h = self.get_max_crop(exp_fold=exp_fold)
        pred_label_dir = os.path.join(exp_fold, 'labels')
        crops_dir = os.path.join(exp_fold, 'crops')
        assert os.path.isdir(crops_dir), self.assert_log(f"'crops_dir':{crops_dir} is not a valid dirpath.")
        images = glob(os.path.join(self.yolo_data_root, 'tiles', '*', 'images', '*.png'))
        fnames = [os.path.basename(file).split('.')[0] for file in images]
        pred_labels = glob(os.path.join(pred_label_dir, '*.txt'))
        reversed_map_classes = {v:k for k,v in self.params['map_classes'].items()}
        all_w, all_h = np.array(all_w), np.array(all_h)
        max_w = np.percentile(all_w, self.crop_percentile)
        max_h = np.percentile(all_h, self.crop_percentile)
        max_size = max(max_w, max_h)
        X_C, Y_C = max_size/2, max_size/2

        # for each pred, look for corresponding image
        for pred_lbl in tqdm(pred_labels, desc='Center cropping'):
            lbl_fn = os.path.basename(pred_lbl).split('.')[0]
            try:
                idx = fnames.index(lbl_fn)
            except:
                raise Exception(f"Index not found for '{lbl_fn} in {fnames}")
            corr_img = images[idx]
            image = cv2.imread(corr_img)
            W,H = image.shape[:2]
            with open(pred_lbl, 'r') as f:
                rows = f.readlines()
            
            # for each obj, create a new cropped image:
            for i, row in enumerate(rows):
                new_image = np.zeros(shape=(int(max_size*W), int(max_size*H), 3))
                clss, x_c, y_c, w, h = self._get_objs_from_row_txt_label(row=row)
                x_min, x_max = int((x_c-w/2)*W), int((x_c+w/2)*W)
                y_min, y_max = int((y_c-h/2)*H), int((y_c+h/2)*H)
                w_old, h_old = x_max-x_min, y_max-y_min
                x_min_new, x_max_new = int(X_C*W - w_old/2), int(X_C*W + w_old/2)
                y_min_new, y_max_new = int(Y_C*H - h_old/2), int(Y_C*H + h_old/2)
                
                if (x_max-x_min) > int(max_size*W) or (y_max-y_min)>int(max_size*H): 
                    fp = os.path.join(crops_dir, reversed_map_classes[clss], f"{lbl_fn}_crop{i}.jpg")
                    self.skipped_crops.append(fp)
                    continue
                if w < self.min_w_h or h < self.min_w_h: 
                    fp = os.path.join(crops_dir, reversed_map_classes[clss], f"{lbl_fn}_crop{i}.jpg")
                    self.skipped_crops.append(fp)
                    continue

                if (x_max_new - x_min_new) != (x_max-x_min): 
                    x_max_new -= (x_max_new - x_min_new) - (x_max-x_min)
                if (y_max_new - y_min_new )!= (y_max-y_min): 
                    y_max_new -= (y_max_new - y_min_new) - (y_max-y_min)

                new_image[y_min_new:y_max_new, x_min_new:x_max_new] = image[y_min:y_max, x_min:x_max]
                os.makedirs(os.path.join(crops_dir, reversed_map_classes[clss]), exist_ok=True)
                fp = os.path.join(crops_dir, reversed_map_classes[clss], f"{lbl_fn}_crop{i}.jpg")
                cv2.imwrite(fp, new_image)

        return
    

    def show_train_data(self, images:torch.Tensor, pred_lbl:torch.Tensor,
                  gt_lbl:torch.Tensor, n_epoch:int = None, ncols:int=2)->None:
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
        fig.savefig('img_cnn_preds.png')
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
        # plt.show()
        fig.savefig(f"img_cnn_metrics.png")
        plt.close()
        return


    def train_model(self)->torch.Tensor:
        """ Trains the VGG model. """
        func_n = self.train_model.__name__

        # set starting values

        model =  self.model
        # criterion = self.criterion
        # optimizer = self.optimizer
        since = time.time() # get start time
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.format_msg(msg=f"Model is on: {next(model.parameters()).device}", func_n=func_n)
        model.to(self.device)   # move model to gpu
        criterion = criterion.to(self.device)

        # TRAINING LOOP
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        for epoch in range(self.epochs):

            # 1) TRAIN for each epoch:
            self.format_msg(msg=f"⏳ Epoch {epoch}/{self.epochs}", func_n=func_n)
            print("⏳ Epoch {}/{}".format(epoch, self.epochs))
            print('-' * 10)
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            model.train(True)
            # for each batch in train loaders:
            for i, data in enumerate(tqdm(self.loaders['train'])):
                inputs, labels = data
                t_batch = labels.shape[0]
                inputs = inputs.to(self.device)     # move images to gpu
                labels =  labels.to(self.device)    # move labels to gpu
                optimizer.zero_grad()               # zero out gradient
                model.zero_grad()
                outputs = model(inputs)             # run input through model and get output
                _, preds = torch.max(outputs.data, 1, keepdim=True)         # get pred from logits
                _, true_classes = torch.max(labels.data, 1, keepdim=True)   # get true class 
                # show data
                if i==0:  self.show_train_data(images=inputs, pred_lbl=preds, gt_lbl=true_classes, n_epoch=epoch)
                weights_before = list(model.parameters())[0].clone()
                loss = criterion(outputs, labels.data)     # evaluate model
                loss.backward()                            # update loss (graph)
                optimizer.step()                           # update optimizer
                weights_after = list(model.parameters())[0].clone()
                # check that weights are being updated
                if epoch==0: assert not {torch.equal(weights_before.data, weights_after.data)}, self.assert_log(f"Weights are not being updated.", func_n=func_n)
                loss_train += loss.data                    # update mean loss 
                acc_train += (torch.sum(preds == true_classes).cpu().numpy() / t_batch) 
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


def eval_model(model, dataloader_cls, 
               dataloaders, criterion)->None:
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
    avg_loss = loss_test / (k+1) #dataloader_cls.valset_size
    avg_acc = acc_test / (k+1) #dataloader_cls.valset_size
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

    return


def infere(model, dataloader_cls, dataloaders, criterion)->None:
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
    avg_loss = loss_test / (k+1) #dataloader_cls.valset_size
    avg_acc = acc_test / (k+1) #dataloader_cls.valset_size
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

    return
    