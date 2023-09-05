import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import utils_yolo, utils_manager
import time
import datetime
from typing import Literal, List
import yaml
from glob import glob
# from loggers import get_logger
from decorators import log_start_finish
# from profiler import Profiler
from configurator import Configurator
from abc import ABC, abstractmethod
from PIL import Image
import random
from crossvalidation import KCrossValidation
from utils import get_config_params
import numpy as np

class YOLOBase(Configurator):

    def __init__(self, 
                 config_yaml_fp:str,
                )-> None:

        super().__init__()
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='yolo_trainer')
        self.map_classes = self.params["map_classes"]
        self.n_classes = len(self.map_classes)
        self.data_folder = os.path.join(self.params["dst_root"], self.params['task'], 'tiles')
        self.tile_size = self.params["tiling_shape"]
        self.batch_size = self.params["batch_size"]
        self.epochs = self.params["epochs"]
        self.repository_dir = self.params["repo"]
        self.yolov5dir = self.params["yolov5dir"]
        self.save_features = self.params["save_features"]
        self.workers = self.params["workers"]
        self.device = self.params["device"]
        self.single_cls = self.params["single_cls"]
        self.dataset = self.params["datasource"]
        self.weights = self.params["yolov5_weights"]
        self.tot_kfolds = self.params["crossvalid_tot_kfolds"]
        self.cur_kfold = self.params["crossvalid_cur_kfold"]
        self.crossvalidator = KCrossValidation(data_root=os.path.dirname(self.data_folder), dataset=self.dataset, k=self.tot_kfolds) if self.tot_kfolds is not None else None
        self.crossvalidation = False if self.crossvalidator is None else True
        self.note = self.params["note"]
        self._class_name = self.__class__.__name__

        self.image_size = self.get_image_size()

        self._parse_args()
        self.add_attributes()

        return
    
    def _parse_args(self): 
        func_n = self._parse_args.__name__
        base_msg = f"{self._class_name}.{func_n}: "
        ALLOWED_SLIDE_FORMATS = ['svs', 'tif']
        ALLOWED_LABEL_FORMATS = ['json']
        ALLOWED_DATA_SOURCES = ['muw', 'tcd', 'hubmap']
        ALLOWED_TASKS = ['segmentation', 'detection']
        ALLOWED_STAINS = ['pas', 'h&e', 'sfog']
        assert (self.tot_kfolds is None and self.cur_kfold is None) or  (self.tot_kfolds is not None and self.cur_kfold is not None) , self.log.error(ValueError(f"{self._class_name}.{'_parse'}: self.tot_kfolds:{self.tot_kfolds} and self.cur_kfold:{self.cur_kfold}, but either they're both None or they both are not None."))
        assert self.cur_kfold < self.tot_kfolds, self.log.error(ValueError(f"{self._class_name}.{'_parse'}: self.tot_kfolds:{self.tot_kfolds} and self.cur_kfold:{self.cur_kfold}, but self.cur_kfold should be < self.tot_kfolds. Please start indexing from 0."))
        assert os.path.isdir(self.params['src_root']), self.log.error(base_msg + f"Repo:{self.params['repo']} is not a valid dirpath.")
        assert os.path.isdir(self.params['dst_root']), self.log.error(base_msg + f"Repo:{self.params['repo']} is not a valid dirpath.")
        assert os.path.isdir(self.params['repo']), self.log.error(base_msg + f"Repo:{self.params['repo']} is not a valid dirpath.")
        assert isinstance(self.map_classes, dict), self.log.error(base_msg+f"Map_classes:{self.map_classes} should be a dict.")
        assert 0 in self.map_classes.values(), self.log.error(base_msg+f"Map_classes:{self.map_classes} should be zero indexed.")
        assert self.params['slide_format'] in ALLOWED_SLIDE_FORMATS, self.log.error(base_msg+f"'slide_format':{self.params['slide_format']} should be one of {ALLOWED_SLIDE_FORMATS}")
        assert self.params['label_format'] in ALLOWED_LABEL_FORMATS, self.log.error(base_msg+f"'label_format':{self.params['label_format']} should be one of {ALLOWED_LABEL_FORMATS}")
        assert self.params['datasource'] in ALLOWED_DATA_SOURCES, self.log.error(base_msg+f"'datasource':{self.params['datasource']} should be one of {ALLOWED_DATA_SOURCES}")
        assert self.params['task'] in ALLOWED_TASKS, self.log.error(base_msg+f"'task':{self.params['task']} should be one of {ALLOWED_TASKS}")
        assert self.params['stain'] in ALLOWED_STAINS, self.log.error(base_msg+f"'stain':{self.params['stain']} should be one of {ALLOWED_STAINS}")
        assert isinstance(self.params['split_ratio'], list), self.log.error(base_msg+f"'split_ratio':{self.params['split_ratio']} should be type list, but is {type(self.params['split_ratio'])}")
        assert (len(self.params['split_ratio']) == 3 or len(self.params['split_ratio']) == 2) and round(np.sum(np.array(self.params['split_ratio'])), 2) == 1.0, self.log.error(base_msg+f"'split_ratio' should be a list of floats with sum 1, but has sum {np.sum(np.array(self.params['split_ratio']))}." )
        assert isinstance(self.params['tiling_shape'], list), self.log.error(base_msg+f"'tiling_shape':{self.params['tiling_shape']} should be type list, but is {type(self.params['tiling_shape'])}")
        assert len(self.params['tiling_shape']) == 2, self.log.error(base_msg+f"'tiling_shape':{self.params['tiling_shape']} should have length = 2, but list, but has length{len(self.params['tiling_shape'])}")
        assert isinstance(self.params['safe_copy'], bool), self.log.error(base_msg+f"'safe_copy':{self.params['safe_copy']} should be boolean but is {type(self.params['safe_copy'])}")
        assert isinstance(self.params['tiling_level'], int), self.log.error(base_msg+f"'tiling_level':{self.params['tiling_level']} should be int, but is type {type(self.params['tiling_level'])}")
        assert isinstance(self.params['image_size'], int), self.log.error(base_msg+f"'image_size':{self.params['image_size']} should be int, but is type {type(self.params['image_size'])}")
        assert isinstance(self.params['resize'], list), self.log.error(base_msg+f"'resize':{self.params['resize']} should be type list, but is {type(self.params['resize'])}")
        assert len(self.params['resize']) == 2, self.log.error(base_msg+f"'resize':{self.params['resize']} should have length = 2, but list, but has length{len(self.params['resize'])}")
        assert isinstance(self.params['yolov5_weights'], str) or self.params['yolov5_weights'] is None, self.log.error(base_msg+f"'yolov5_weights':{self.params['yolov5_weights']} should be either string or None, but is {type(self.params['yolov5_weights'])}")
        if self.params['yolov5_weights'] is not None: assert not (self.params['task']=='detection' and 'seg' in self.params['yolov5_weights']), self.log.error(base_msg+f"'task' is {self.params['task']} but 'yolov5_weights' is {self.params['yolov5_weights']}")
        if self.params['yolov5_weights'] is not None: assert not (self.params['task']=='segmentation' and 'seg' not in self.params['yolov5_weights']), self.log.error(base_msg+f"'task' is {self.params['task']} but 'yolov5_weights' is {self.params['yolov5_weights']}")
        
        return
    

    def add_attributes(self): 
        """ Adds other attributes depending on dataset. """
        
        if self.dataset == 'hubmap':
            self.wsi_images_like = '*.tif'
            self.wsi_labels_like = '*.json'
            self.tile_images_like = '*.png'
            self.tile_labels_like = '*.txt'
        elif self.dataset == 'muw':
            self.wsi_images_like = '*.tif', 
            self.wsi_labels_like = '*_sample?.txt',
            self.tile_images_like = '*sample*.png',
            self.tile_labels_like = '*sample*.txt'
        elif self.dataset == 'tcd':
            self.wsi_images_like = '*.svs', 
            self.wsi_labels_like = '*_sample?.txt',
            self.tile_images_like = '*sample*.png',
            self.tile_labels_like = '*sample*.txt'
        else: 
            raise ValueError(f"Wrong dataset value")

        return
    
    @abstractmethod
    def train(self, weights: str = None) -> None:
        return
    
    
    @abstractmethod
    def _log_data(self):



        return
    
    def _get_train_duration(self, start): 

        end = datetime.datetime.now()
        diff = (end - start)

        diff_seconds = int(diff.total_seconds())
        minute_seconds, seconds = divmod(diff_seconds, 60)
        hours, minutes = divmod(minute_seconds, 60)
        print(f"Train duration: {hours}h {minutes}m {seconds}s")

        return hours, minutes, seconds


    def get_exp_fold(self, exp_fold:str) ->str:
        """ Returns last exp folder from the exp_fold"""
        
        assert os.path.isdir(exp_fold), self.log.error(ValueError(f"{self._class_name}.{'_get_last_fold'}: 'exp_fold' is not a valid dirpath."))

        exps = [subdir for subdir in os.listdir(exp_fold) if (subdir!='exp' and 'exp' in subdir)] # excluding the one w/o number
        nums = [int(exp.replace('exp','')) for exp in exps]
        nums = sorted(nums)

        new_fold = os.path.join(exp_fold, f"exp{nums[-1]+1}")
        self.log.info(f"{self._class_name}.{'get_exp_fold'}: exp folder: {new_fold}.")

        return new_fold
    
    def get_image_size(self):

        tile_fold = self.data_folder
        assert os.path.isdir(tile_fold), f"tile_fold:{tile_fold} is not a valid dirpath."
        images = glob(os.path.join(tile_fold, 'train', 'images', '*.png'))
        assert len(images)>0, f"'images' is empty. No images like {os.path.join(tile_fold, 'train', 'images', '*.png')} "
        file = random.choice(images)
        image = Image.open(file)
        image_size = image.size

        return image_size[0]
    


    def _edit_yaml(self) -> None:
        """ Edits YAML data file from yolov5. """

        self.log.info("⏳ Setting configurations for YOLO: ")
        classes = dict([(value, key) for key, value in self.map_classes.items()])
       
        # raise Exception
        yaml_fp = os.path.join(self.yolov5dir, 'data', 'helical.yaml')
        text = {'path':self.data_folder, 'train': os.path.join(self.data_folder, 'train', 'images'), 'val': os.path.join(self.data_folder, 'val', 'images'), 'test': os.path.join(self.data_folder, 'test', 'images'), 'names':classes}
        with open(yaml_fp, 'w') as f:
            yaml.dump(data = text, stream=f)
        self.log.info(f"✅ YOLO set up completed YOLO ✅ .")

        return yaml_fp


    

    def save_training_data(self, weights:str, start_time:str) -> None:
        """ Saves training data into a json file in the runs folder from YOLO. """

        # get file splitting: 
        if os.path.isdir(self.data_folder.replace('tiles', 'wsi')):
            sets = ['train', 'val', 'test']
            data = {}
            for dirname in sets: 
                dirpath = os.path.join(self.data_folder, dirname, 'images')
                data[dirname] = [file for file in os.listdir(dirpath) if 'DS' not in file ]
        # print(f"dictionary data: {data}")
        # get training duration:
        end_time = time.time()
        train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        # save info into json file:
        otherinfo_yolo = {'datafolder': self.data_folder, 'data':data, 'classes': self.map_classes, 'epochs': self.epochs, 'duration': train_yolo_duration, 'weights': {weights}}
        utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = os.path.join(self.yolov5dir, 'runs', 'train'))
        self.log.info(f"Training YOLO done ✅ . Training duration: {train_yolo_duration}")

        return
    





def test_YOLOSegmentor(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_hubmap_segm_manager/detection/tiles' if system == 'mac' else r'D:\marco\datasets\slides\detection\tiles'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    # map_classes = {'glomerulus':0}  #{'Glo-healthy':1, 'Glo-unhealthy':0}
    # save_features = True
    # tile_size = 512 
    # batch_size=2 if system == 'mac' else 2
    # epochs=5
    map_classes =  {'glomerulus':0} # {'Glo-healthy':1, 'Glo-unhealthy':0}
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=10
    segmentor = YOLOBase(data_folder=data_folder, 
                            repository_dir=repository_dir,
                            yolov5dir=yolov5dir,
                            map_classes=map_classes,
                            tile_size = tile_size,
                            batch_size=batch_size,
                            epochs=epochs,
                            workers=workers,
                            device=device,
                            save_features=save_features)
    segmentor.train()

    return




if __name__ == '__main__':
    
    test_YOLOSegmentor()
