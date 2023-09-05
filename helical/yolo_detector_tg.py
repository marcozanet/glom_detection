import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import Any, List
import time
import datetime
from loggers import get_logger
from typing import Literal, List
import yaml
from glob import glob
import torch
from utils import get_config_params
import cv2
import numpy as np
import matplotlib.pyplot as plt


class YOLO_Detector():

    def __init__(self, 
                 yaml_fp: str,
                 trained_model_weights: str,
                 conf_thres: float,
                 ) -> None: 

        self.log = get_logger()
        self.params = get_config_params(yaml_fp=yaml_fp, config_name='inference')
        self.trained_model_weights = trained_model_weights
        self.conf_thres = conf_thres
        self.image_dir = self.params['input_dir']
        return
    
    # def _parse_args(self):

    #     assert os.path.isdir(self.params['images_dir']), ValueError(f"'images_dir': {self.params['images_dir']} is not a valid dirpath.")
    #     assert os.path.isdir(self.params['yolov5dir']), ValueError(f"'yolov5dir': {self.params['yolov5dir']} is not a valid dirpath.")
        
    #     return

    def __call__(self) -> Any:

        # 1) parse args: 
        # parse args 
        # 2) infere:
        self.infere()
        self.log.info(f"WEEEEEEERE DOOOOONEEEE YEG")
        # 3) create masks:
        if self.params['create_masks'] is True:
            self._convert_labels2masks()
        return


    def infere(self) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        os.chdir(self.params['yolov5dir'])
        if self.params['device'] == 'gpu':
            device='cuda:0' if torch.cuda.is_available() else 'cpu'

        # 2) define command:
        command = f"python detect.py --source {self.image_dir} --weights {self.trained_model_weights}  --data data/helical.yaml --device {device}"
        if self.params['augment'] is True:
            command += " --augment"
        if self.params['save_imgs'] is False:
            command += " --nosave"
        if self.params['save_txt'] is True: 
           command +=" --save-txt"
        if self.params['imgsz'] is not None: 
           command +=" --save-txt"
        
        # if self.params['save_crop'] is True: 
        #    command +=" --save-crop"
        if self.conf_thres is not None:
            command += f" --conf-thres {self.conf_thres}" 
        if self.params['output_dir'] is not None:
            command += f" --project {os.path.split(self.params['output_dir'])[0]}" 
        if self.params['output_dir'] is not None:
            command += f" --name {os.path.split(self.params['output_dir'])[1]}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.params['repository_dir'])
        self.log.info(f"Inference YOLO done ✅ .")

        get_last_fold = lambda fold: max([os.path.join(fold, subfold) for subfold in os.listdir(fold)], key=os.path.getmtime)
        self.output_dir = get_last_fold(os.path.join(self.params['yolov5dir'], 'runs', 'detect')) if self.params['output_dir'] is None else self.params['output_dir']

        return
    
    
    def _convert_labels2masks(self):

        path_to_labels, path_to_images = os.path.join(self.output_dir, 'labels'), self.params['input_dir']
        self.labels = [os.path.join(path_to_labels, file) for file in os.listdir(path_to_labels) if '.txt' in file and 'DS' not in file]
        self.images = [os.path.join(path_to_images, file) for file in os.listdir(path_to_images) if '.png' in file and 'DS' not in file]
        # assert len(self.labels)>0, f"'ouput_dir':{path_to_labels} does not contain '.txt' files: {os.listdir(path_to_labels)}"
        # assert len(self.images)>0, f"'ouput_dir':{path_to_images} does not contain '.png' files: {os.listdir(path_to_images)}"
        ex_image = self.images[0]
        self.image_shape = cv2.imread(ex_image).shape

        if self.params['task']=='detection': 
            if self.params['create_masks'] is True:
                for label in self.labels:
                    self._make_rect_mask(label_fp=label)
        else:
            raise NotImplementedError()
        return 

    def _make_rect_mask(self, label_fp:str):

        # for all files to process
        label = label_fp # TODO change to func!!
        mask = np.zeros(shape = self.image_shape[:2])

        with open(label, 'r') as f: # read label
            objects = f.readlines()

        # create object_mask and class_mask:
        obj_mask, clss_mask = np.zeros(shape = self.image_shape[:2]), np.zeros(shape = self.image_shape[:2])
        for i, obj in enumerate(objects, start=1):
            class_ = int(obj[0]) 
            coords = [float(val) for j,val in enumerate(obj.split(' ')) if j>0]
            xc, yc, w, h = [int(coord*mask.shape[0]) if i%2==0 else int(coord*mask.shape[1]) for i,coord in enumerate(coords)]
            x0, y0, x1, y1 = int(xc-w/2), int(yc-h/2), int(xc+w/2), int(yc+h/2)
            obj_val, clss_val = i, class_ + 1
            obj_mask = cv2.rectangle(img=obj_mask, pt1=(x0,y0), pt2=(x1,y1), color=obj_val, thickness=-1)
            clss_mask = cv2.rectangle(img=clss_mask, pt1=(x0,y0), pt2=(x1,y1), color=clss_val, thickness=-1)
    
        if self.params['show_mask'] is True: # show masks
            plt.imshow(obj_mask)
            plt.show()
            plt.imshow(clss_mask)
            plt.show()

        # save in dirs:
        obj_mask_dir = os.path.join(self.output_dir, 'object_masks') 
        os.makedirs(obj_mask_dir, exist_ok=True)
        obj_mask_fp = os.path.join(obj_mask_dir, os.path.basename(label_fp).split('.')[0] + '.png' )
        cv2.imwrite(obj_mask_fp, obj_mask)
        clss_mask_dir = os.path.join(self.output_dir, 'class_masks')
        os.makedirs(clss_mask_dir, exist_ok=True)
        class_mask_fp = os.path.join(clss_mask_dir, os.path.basename(label_fp).split('.')[0] + '.png' )
        cv2.imwrite(class_mask_fp, clss_mask)

        return
    


if __name__ == '__main__': 
    yaml_fp = '/Users/marco/yolo/code/helical/tg_config_test.yaml'
    trained_model_weights = '/Users/marco/yolov5/runs/train/exp140/weights/best.pt'
    detector = YOLO_Detector(yaml_fp=yaml_fp, 
                             trained_model_weights=trained_model_weights,
                             conf_thres=0.445)
    detector()