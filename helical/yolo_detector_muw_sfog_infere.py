import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import List
import time
import datetime
from loggers import get_logger
from typing import Literal, List
import yaml
from glob import glob


class YOLO_Inferer_Detector():

    def __init__(self, 
                images_dir: str,
                weights: str,
                yolov5dir: str,
                repository_dir:str,
                # map_classes: dict = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3},
                # system = 'mac',
                augment: bool = False,
                # tile_size = 512,
                # batch_size = 8,
                # epochs = 3,
                conf_thres = 0.8,
                ) -> None: 

        self.log = get_logger()
        assert isinstance(conf_thres, float) or conf_thres is None, TypeError(f"conf_thres is {type(conf_thres)}, but should be either None or float.")

        self.yolov5dir = yolov5dir
        self.images_dir = images_dir
        self.weights = weights
        # self.map_classes = map_classes
        # self.tile_size = tile_size
        # self.batch_size = batch_size
        # self.epochs = epochs
        self.conf_thres = conf_thres
        self.repository_dir = repository_dir
        self.augment = augment
        # self.system = system


        return
    




    def infere(self, visualize: bool = False, save_txt: bool = False) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        assert os.path.isdir(self.images_dir), ValueError(f"'images_dir': {self.images_dir} is not a valid dirpath.")
        os.chdir(self.yolov5dir)

        # 1) prepare inference:
        # weights_dir = self._prepare_inference(yolo_weights=yolo_weights)

        # 2) define command:
        command = f'python detect.py --source {self.images_dir} --weights {self.weights}  --data data/helical.yaml --device cpu'
        if self.augment is True:
            command += " --augment"
        if save_txt is True: 
           command +=" ---save-txt"
        if visualize is True:
            command +=" --visualize"
        if self.conf_thres is not None:
            command += f" --conf-thres {self.conf_thres}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.repository_dir)
        self.log.info(f"Inference YOLO done ✅ .")

        return



    def _prepare_inference(self, yolo_weights:str = None) -> str:
        """ Prepares inference with YOLO. """

        # get model:
        os.chdir(self.yolov5dir)
        if yolo_weights is None:
            raise NotImplementedError()
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = os.path.dirname(yolo_weights)

        self.log.info(f"Prepared YOLO for inference ✅ .")

        return weights_dir




