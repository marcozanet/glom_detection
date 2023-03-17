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

class YOLOSegmentor(Configurator):

    def __init__(self, 
                data_folder:str,
                yolov5dir: str,
                repository_dir:str,
                map_classes: dict = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3},
                save_features: bool = False,
                tile_size = 512,
                batch_size = 8,
                workers = 1,
                device = None,
                epochs = 3,
                ) -> None: 

        super().__init__()
        # self.log = get_logger()

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
        self.task = 'segmentation'


        self._class_name = self.__class__.__name__

        return
    

    def train(self, weights: str = None) -> None:
        """   Runs the YOLO detection model. """
        class_name = self.__class__.__name__
        
        # 1) prepare training:
        # start_time = time.time()
        yaml_fn = os.path.basename(self._edit_yaml())
        weights = weights if weights is not None else 'yolov5s-seg.pt'
        self.log.info(f"{class_name}.{'train'}: weights:{weights}")
        # self._log_data_pretraining()

        @log_start_finish(class_name=class_name, func_name='train', msg = f" YOLO training:" )
        def do():
            # 2) train:
            self.log.info(f"⏳ Starting YOLO segmentation:")
            os.chdir(self.yolov5dir)
            prompt = f"python segment/train.py --img {self.tile_size} --batch {self.batch_size} --epochs {self.epochs}"
            prompt += f" --data {yaml_fn} --weights {weights} --workers {self.workers}"
            prompt = prompt+f" --device {self.device}" if self.device is not None else prompt 
            self.log.info(f"{class_name}.{'train'}: {prompt}")
            os.system(prompt)
            os.chdir(self.repository_dir)

            # 3) save:
            # self.save_training_data(weights=weights, start_time=start_time)
        
            return
        
        do()

        return  
    
    def _log_data_pretraining(self): 
        """ Logs a bunch of dataset info prior to training. """

        if 'hubmap' in self.data_folder:
            profiler = Profiler(data_root=os.path.dirname(self.data_folder),
                                wsi_images_like = '*.tif', 
                                wsi_labels_like = '*.txt',
                                tile_images_like = '*_*_*.png',
                                tile_labels_like = '*_*_*.txt')
            profiler.log_data_summary()

        else:
            profiler = Profiler(data_root=os.path.dirname(self.data_folder) )
            profiler.log_data_summary()


        return

           

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
    segmentor = YOLOSegmentor(data_folder=data_folder, 
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
