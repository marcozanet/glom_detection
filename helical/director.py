import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from loggers import get_logger
from utils import get_config_params
from tcd_processor import Processor
from tcd_yolo_trainer import YOLO_Trainer 
from tg_inference import YOLO_Inferer
from tg_validate import YOLO_Validator
from cnn_trainer import CNN_Trainer
from cnn_process import CNN_Process
from cnn_validator import CNN_Validator
from cnn_inferer import CNN_Inferer
from cnn_feature_extractor import CNN_FeatureExtractor
from utils import get_config_params
from configurator import Configurator


class Director(Configurator):

    def __init__(self,
                 config_yaml_fp:str,
                 models_yaml_fp:str,
                 ) -> None:
        
        self.log = get_logger()
        self.config_yaml_fp = config_yaml_fp
        self.models_yaml_fp = models_yaml_fp
        self.class_n = self.__class__.__name__
        return
    

    def yolo_process(self):
        processor = Processor(config_yaml_fp=self.config_yaml_fp)
        processor()
        return


    def yolo_train(self):
        detector = YOLO_Trainer(config_yaml_fp=self.config_yaml_fp)
        detector.train()
        return
    
    
    def yolo_infere(self):
        inferer = YOLO_Inferer(config_yaml_fp=self.config_yaml_fp, models_yaml_fp=self.models_yaml_fp)
        inferer()
        return

        

    def yolo_infere_trainvaltest(self):
        func_n = self.yolo_infere_trainvaltest.__name__

        base_msg = f"{self.class_n}.{func_n} "
        self.infere_params = get_config_params(self.config_yaml_fp, 'inference')
        datasets = ['test', 'val', 'train']
        assert os.path.split(os.path.dirname(self.infere_params['input_dir']))[1] in datasets, f"dirname is {os.path.dirname(self.infere_params['input_dir'])} but should be train, val or test."
        change_set = lambda fp, _set: os.path.join(os.path.dirname(os.path.dirname(fp)), _set, 'images')

        assert self.infere_params['save_crop'] is True
        
        for _set in datasets: 
            self.log.info(base_msg+f"Infering on {_set}.")
            print("-"*20)
            images_dir = change_set(self.infere_params['input_dir'], _set)
            inferer = YOLO_Inferer(config_yaml_fp=self.config_yaml_fp, models_yaml_fp=self.models_yaml_fp)
            inferer.input_dir = images_dir
            inferer()
        self.log.info(base_msg+f"✅ Infered on train, val, test. Infere out folds in {self.infere_params['input_dir']} ")

        return
    
    def cnn_process(self, mode:str):
        cnn_processor = CNN_Process(config_yaml_fp=self.config_yaml_fp)
        cnn_processor(mode)
        return

    def cnn_train(self):
        cnn_trainer = CNN_Trainer(config_yaml_fp=self.config_yaml_fp)
        cnn_trainer()
        return

    def cnn_validate(self):
        cnn_validator = CNN_Validator(config_yaml_fp=self.config_yaml_fp)
        cnn_validator()
        return
    
    def cnn_extract_features(self):
        cnn_feature_extractor = CNN_FeatureExtractor(config_yaml_fp=self.config_yaml_fp)
        cnn_feature_extractor()
        return
    
    def cnn_process_inference(self):
        self.cnn_process(mode='inference')


        return
    
    def cnn_infere(self):
        cnn_inferer = CNN_Inferer(config_yaml_fp=self.config_yaml_fp)
        cnn_inferer()
        return
    

    def run_pipeline(self):

        self.yolo_process()
        self.yolo_train()
        # self.yolo_validate() # choose best model and save
        self.yolo_infere_trainvaltest()
        self.cnn_process()
        self.cnn_train()

        return
 



if __name__ == '__main__':
    from sys import platform 
    if platform == 'darwin':
        config_yaml_fp = 'config_tcd.yaml'
        models_yaml_fp = 'config_saved_models.yaml'
    else:
        config_yaml_fp = 'config_tcd_windows.yaml'
        models_yaml_fp = 'config_saved_models_windows.yaml'
    director = Director(config_yaml_fp=config_yaml_fp, models_yaml_fp=models_yaml_fp)
    director.cnn_infere()