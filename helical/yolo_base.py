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
from profiler import Profiler
from configurator import Configurator
from abc import ABC, abstractmethod


class YOLOBase(Configurator):

    def __init__(self, 
                 dataset:Literal['hubmap', 'muw'],
                 data_folder:str,
                 yolov5dir: str,
                 repository_dir:str,
                 map_classes: dict,
                 tile_size: int,
                 batch_size: int,
                 workers:int,
                 epochs: int,
                 save_features: bool = False,
                 device = None) -> None: 

        super().__init__()

        self.map_classes = map_classes
        self.data_folder = data_folder
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.repository_dir = repository_dir
        self.yolov5dir = yolov5dir
        self.save_features = save_features
        self.workers = workers
        self.device = device
        self.dataset = dataset
        self.n_classes = len(self.map_classes)
        if dataset not in data_folder:
            self.log.warning(f"{self._class_name}.{'__init__'}: 'dataset' is '{dataset}', but 'data_folder' is '{data_folder}'")
        self._class_name = self.__class__.__name__


        self.add_attributes()

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
        else: 
            raise ValueError(f"Wrong dataset value")

        return
    
    @abstractmethod
    def train(self, weights: str = None) -> None:
        return
    
    
    @abstractmethod
    def _log_data(self):



        return


    def get_exp_fold(self, exp_fold:str) ->str:
        """ Returns last exp folder from the exp_fold"""
        
        assert os.path.isdir(exp_fold), self.log.error(ValueError(f"{self._class_name}.{'_get_last_fold'}: 'exp_fold' is not a valid dirpath."))

        exps = [subdir for subdir in os.listdir(exp_fold) if (subdir!='exp' and 'exp' in subdir)] # excluding the one w/o number
        nums = [int(exp.replace('exp','')) for exp in exps]
        nums = sorted(nums)

        new_fold = os.path.join(exp_fold, f"exp{nums[-1]}")
        self.log.info(f"{self._class_name}.{'get_exp_fold'}: exp folder: {new_fold}.")

        self.exp_fold = new_fold

        return new_fold
    


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
