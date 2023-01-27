import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import utils_yolo, utils_manager
from typing import List
from model import GlomModel
from dataloader import get_loaders
import pytorch_lightning as pl
import predict_unet as pu
from processor_tile import TileProcessor
from processor_wsi import WSI_Processor
import time
import datetime
from loggers import get_logger
from typing import Literal, List
from manager import Manager
import yaml


class YOLODetector():

    def __init__(self, 
                data_folder:str,
                yolov5dir: str,
                yolodir:str,
                map_classes: dict = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3},
                # system = 'mac',
                tile_size = 512,
                batch_size = 8,
                epochs = 3,
                conf_thres = None,
                ) -> None: 

        self.log = get_logger()
        assert isinstance(conf_thres, float) or conf_thres is None, TypeError(f"conf_thres is {type(conf_thres)}, but should be either None or float.")

        self.map_classes = map_classes
        self.data_folder = data_folder
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.conf_thres = conf_thres
        self.yolodir = yolodir
        self.yolov5dir = yolov5dir
        # self.system = system


        return
    




    def train(self, weights: str = None) -> None:
        """   Runs the YOLO detection model. """
        
        # 1) prepare training:
        start_time = time.time()
        yaml_fn = os.path.basename(self._edit_yaml())
        weights = weights if weights is not None else 'yolov5s.pt'

        # 2) train:
        self.log.info(f"⏳ Start training YOLO:")
        os.chdir(self.yolov5dir)
        os.system(f' python train.py --img {self.tile_size} --batch {self.batch_size} --epochs {self.epochs} --data {yaml_fn} --weights {weights}')
        os.chdir(self.yolodir)

        # 3) save:
        end_time = time.time()
        train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        otherinfo_yolo = {'data': self.data_folder, 'classes': self.map_classes, 'epochs': self.epochs, 'duration': train_yolo_duration}
        utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = '/Users/marco/yolov5/runs/train')
        self.log.info(f"Training YOLO done ✅ . Training duration: {train_yolo_duration}")

        return  


    def _prepare_testing(self, yolo_weights:bool = False ) -> str:
        """ Prepares testing with YOLO. """

        # get model
        os.chdir(self.yolov5dir)
        if yolo_weights is False:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = yolo_weights

        self.log.info(f"Prepared YOLO testing ✅ .")

        return weights_dir

    
    def test(self, val_augment:bool = False) -> None:
        """ Tests YOLO on the test set and returns performance metrics. """
        
        # 1) prepare testing:
        weights_dir = self._prepare_testing()

        # 2) define command:
        command = f'python val.py --task test --weights {weights_dir} --data data/hubmap.yaml --device cpu'
        if val_augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf_thres {self.conf_thres}" 

        # 3) test (e.g. validate):
        self.log.info(f"Start testing YOLO: ⏳")
        os.system(command)
        os.chdir(self.yolodir)
        self.log.info(f"Testing YOLO done ✅ .")

        return


    def _prepare_inference(self, yolo_weights:str = None) -> str:
        """ Prepares inference with YOLO. """

        # get model:
        os.chdir(self.yolov5dir)
        if yolo_weights is None:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = os.path.dirname(yolo_weights)

        self.log.info(f"Prepared YOLO for inference ✅ .")

        return weights_dir


    def infere(self, images_dir: str, yolo_weights:str = None, infere_augment:bool = False) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        assert os.path.isdir(images_dir), ValueError(f"'images_dir': {images_dir} is not a valid dirpath.")

        # 1) prepare inference:
        weights_dir = self._prepare_inference(yolo_weights=yolo_weights)

        # 2) define command:
        command = f'python detect.py --source {images_dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --save-txt '
        if infere_augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf_thres {self.conf_thres}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.yolodir)
        self.log.info(f"Inference YOLO done ✅ .")

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

    # def _infere_for_UNet(self, infere_augment:bool = False) -> None:
    #     """ Applies inference with YOLO on train, val, test sets. """

    #     # 1) prepare inference:
    #     weights_dir = self._prepare_inference()

    #     # 2) infere on train, val, test sets:
    #     dirs = [self.train_dir, self.val_dir, self.test_dir]
    #     dirs = [os.path.join(dir, 'images') for dir in dirs]
    #     dirnames = ['train', 'val', 'test']
    #     for dirname, image_dir in zip(dirnames, dirs):
    #         command = f'python detect.py --source {image_dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --save-txt '
    #         if infere_augment is True:
    #             command += " --augment"
    #         self.log.info(f"Start inference YOLO on {dirname}: ⏳")
    #         os.system(command)
    #         self.log.info(f"Inference YOLO done on {dirname} ✅ .")

    #     os.chdir(self.yolodir)

    #     return





def test_YOLODetector(): 

    system = 'mac'
    yolodir = '/Users/marco/yolo/code/helical' if system == 'mac' else 'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else 'C:\marco\yolov5'

    data_folder = '/Users/marco/Downloads/try_train/detection/tiles'
    map_classes = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3}
    tile_size = 512
    batch_size=8
    epochs=2
    conf_thres=0.7
    detector = YOLODetector(data_folder=data_folder,
                            yolodir=yolodir,
                            yolov5dir=yolov5dir,
                            map_classes=map_classes,
                            tile_size = tile_size,
                            batch_size=batch_size,
                            epochs=epochs,
                            conf_thres=conf_thres)
    detector.train()

    return
        


if __name__ == '__main__':
    
    test_YOLODetector()

    