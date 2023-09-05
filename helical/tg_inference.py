import os, sys
from typing import Callable, Any
import numpy as np
from yolo_detector_tg import YOLO_Detector
from utils import get_config_params, get_trained_model_weight_paths



class YOLO_Inferer(): 

    def __init__(self, 
                 config_yaml_fp:str,
                 models_yaml_fp:str) -> None:
        
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='inference')
        self.trained_models = get_trained_model_weight_paths(yaml_fp=models_yaml_fp)
        self.input_dir = self.params['input_dir']
        print(self.trained_models)

        self._choose_weights_()
        
        return
    
    
    def _parse_args(self)->None:
        """ Parses class arguments. """

        ALLOWED_MODELS = ['yolo', 'unet']
        ALLOWED_TASKS = ['segmentation', 'detection']
        assert isinstance(self.params['classify'], bool), f"'classify' param should be a boolean. "
        assert self.params['model'] in ALLOWED_MODELS, f"'model' param should be one of {ALLOWED_MODELS}"
        assert self.params['task'] in ALLOWED_TASKS, f"'task' param should be one of {ALLOWED_TASKS}"
        assert not (self.params['model']=='unet' and self.params['task']=='detection'), f"'model'='unet', but 'task'='detection'. Please select either yolo for detection or unet for segmentation."
        assert os.path.isdir(self.params['input_dir']), f"'input_dir':{self.params['input_dir']} is not a valid dir path."

        return
    
    
    def _choose_weights_(self)->None:
        """ Depending on params configuration, it returns a function to do inference. """

        classification = 'w_classification' if self.params['classify'] is True else 'wo_classification'
        self.weights= self.trained_models[self.params['model']][self.params['task']][classification]['weights']
        self.conf_thres= self.trained_models[self.params['model']][self.params['task']][classification]['conf_thres']

        return 

    def _infere_(self)->None:
        """ Inferes on the images using the config trained model. """

        if self.params['model'] == 'yolo':
            detector = YOLO_Detector(yaml_fp=self.config_yaml_fp,
                                     trained_model_weights=self.weights,
                                     conf_thres=self.conf_thres)
            detector.image_dir = self.input_dir
            detector()
        else: 
            raise NotImplementedError()

        return
    

    def __call__(self)->None:

        self._parse_args()
        self._choose_weights_()
        self._infere_()

        return
    

if __name__== '__main__':
    config_yaml_fp = '/Users/marco/yolo/code/helical/tg_config_test.yaml'
    models_yaml_fp = '/Users/marco/yolo/code/helical/trained_models.yaml'
    inferer = YOLO_Inferer(config_yaml_fp=config_yaml_fp, models_yaml_fp=models_yaml_fp)
    inferer()
