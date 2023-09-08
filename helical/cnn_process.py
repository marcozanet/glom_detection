import os, shutil
from glob import glob
from typing import Any
from tqdm import tqdm 
import numpy as np 
import cv2
from cnn_trainer_base import CNN_Trainer_Base
from cnn_assign_class_crop_new import CropLabeller
from cnn_splitter import CNNDataSplitter


class CNN_Process(CNN_Trainer_Base):

    def __init__(self, config_yaml_fp: str) -> None:
        super().__init__(config_yaml_fp)
        return


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
    
    
    def prepare_data(self, mode:str)->None:
        """ Prepares data for CNN training: puts all images in the same folder 
            (regardless of trainset, valset, testset) and gets the dataloader 
            to be used by the model."""
        func_n = self.prepare_data.__name__
        msg_base = f"{self.class_name}.{func_n}: "
        self.log.info(msg_base + f"⏳ Preparing data for CNN training:")

        assert mode in ['train', 'val', 'inference'], self.assert_log(f"'mode':{mode} should be one of ['train','val','inference']")


        # Assign true classes back to crops out of yolo:
        # assert 'false_positives' in self.map_classes.keys(), f"'false_positives' missing in 'map_classes'. "
        cnn_root = os.path.join(self.cnn_data_fold, 'cnn_dataset')
        if mode in ['train', 'val']: self.log.info(msg_base + f"⏳ Labelling crops out of YOLO:") 
        else: self.log.info(msg_base + f"⏳ Cropping images out YOLO:") 

        for exp_fold in self.yolo_exp_folds:
            # 1) create crops:
            self.center_crop(exp_fold=exp_fold)
            if mode=='inference': # will only infere on the last fold
                self.log.info(msg_base + f"✅ Cropped images.")
                self._move_inference_crops(exp_fold=exp_fold)
                return
            # 2) assign each crop the correct label:
            if mode in ['train', 'val']:
                labeller = CropLabeller(self.config_yaml_fp, exp_fold=exp_fold, skipped_crops=self.skipped_crops)
                labeller()

        self.log.info(msg_base + f"✅ Labelled crops.")
        # Creating Dataset and splitting into train, val, test:
        self.log.info(msg_base + f"⏳ Creating train,val,test sets:")
        cnn_splitter = CNNDataSplitter(src_folds=self.yolo_exp_folds, map_classes=self.map_classes, yolo_root=self.yolo_data_root, 
                                        dst_root=cnn_root, treat_as_single_class=self.treat_as_single_class)
        cnn_splitter()
        cnn_dataset_fold = os.path.join(self.cnn_data_fold, 'cnn_dataset')
        self.log.info(msg_base + f"✅ Created train,val,test sets.")

        return cnn_dataset_fold
    
    
    def _move_inference_crops(self, exp_fold:str):
        """ Copies from runs/detect folder images to new test set. """

        func_n = self._move_inference_crops.__name__
        old_dir = os.path.join(exp_fold, 'crops')
        crops = glob(os.path.join(old_dir, '*', '*.jpg'))
        assert len(crops)>0, self.assert_log(f"'crops' is empty. Path like: {os.path.join(old_dir, '*', '*.jpg')}", func_n=func_n)
        new_dir = os.path.join(self.cnn_data_fold, 'cnn_dataset', 'test')
        assert os.path.isdir(new_dir), self.assert_log(f"'new_dir':{new_dir} is not a valid dirpath.", func_n=func_n)
        shutil.rmtree(new_dir) # cleaning from subfolders created by crossvalidation
        os.makedirs(new_dir)

        for src in tqdm(crops, 'Copying crops to infere to test fold'): 
            dst = os.path.join(new_dir, os.path.basename(src))
            shutil.copy(src, dst)

        return
    


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
        path_like = os.path.join(self.yolo_data_root, 'tiles', '*', 'images', '*.png')
        images = glob(path_like)
        assert len(images)>0, self.assert_log(f"'images' is empty. Path like:{path_like}", func_n=func_n)
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
    
    def __call__(self, mode='train'):
        self.prepare_data(mode=mode)
        return 
    