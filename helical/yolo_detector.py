import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import utils_yolo, utils_manager
from typing import List
import time
import datetime
from loggers import get_logger
from typing import Literal, List
import yaml
from glob import glob


class YOLODetector():

    def __init__(self, 
                data_folder:str,
                yolov5dir: str,
                repository_dir:str,
                map_classes: dict = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3},
                # system = 'mac',
                save_features: bool = False,
                tile_size = 512,
                batch_size = 8,
                workers = 1,
                device = None,
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
        self.repository_dir = repository_dir
        self.yolov5dir = yolov5dir
        self.save_features = save_features
        self.workers = workers
        self.device = device
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
        prompt = f'python train.py --img {self.tile_size} --batch {self.batch_size} --epochs {self.epochs} --data {yaml_fn} --weights {weights} --workers {self.workers}'
        prompt = prompt+f" --device {self.device}" if self.device is not None else prompt 
        os.system(prompt)
        os.chdir(self.repository_dir)

        # 3) save:
        self.save_training_data(weights=weights, start_time=start_time)

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
        command = f'python detect.py --source {images_dir} --weights {weights_dir} --data data/helical.yaml --device cpu --save-txt '
        if infere_augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf_thres {self.conf_thres}" 
        if self.save_features is True:
            command += f" --visualize" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.repository_dir)
        self.log.info(f"Inference YOLO done ✅ .")

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
        os.chdir(self.repository_dir)
        self.log.info(f"Testing YOLO done ✅ .")

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





def test_YOLODetector(): 

    system = 'mac'
    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/Downloads/train_20feb23/tiles' if system == 'mac' else r'D:\marco\datasets\muw\detection\tiles'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1}
    save_features = True
    tile_size = 512
    batch_size=4
    epochs=1
    conf_thres=0.7
    detector = YOLODetector(data_folder=data_folder,
                            repository_dir=repository_dir,
                            yolov5dir=yolov5dir,
                            map_classes=map_classes,
                            tile_size = tile_size,
                            batch_size=batch_size,
                            epochs=epochs,
                            workers=workers,
                            device=device,
                            save_features=save_features,
                            conf_thres=conf_thres)
    detector.train()

    return
        



if __name__ == '__main__':
    
    test_YOLODetector()
