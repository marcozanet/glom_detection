import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import List
import time
import datetime
from loggers import get_logger
from typing import Literal, List
import yaml
from glob import glob


class YOLODetector():

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

        # self.map_classes = map_classes
        self.images_dir = images_dir
        # self.tile_size = tile_size
        # self.batch_size = batch_size
        # self.epochs = epochs
        self.conf_thres = conf_thres
        self.repository_dir = repository_dir
        self.yolov5dir = yolov5dir
        self.weights = weights
        self.augment = augment
        # self.system = system


        return
    




    def infere(self) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        assert os.path.isdir(self.images_dir), ValueError(f"'images_dir': {self.images_dir} is not a valid dirpath.")
        
        os.chdir(self.yolov5dir)

        # 1) prepare inference:
        # weights_dir = self._prepare_inference(yolo_weights=yolo_weights)

        # 2) define command:
        command = f'python detect.py --source {self.images_dir} --weights {self.weights} --visualize --data data/helical.yaml --device cpu --save-txt '
        if self.augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf-thres {self.conf_thres}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.repository_dir)
        self.log.info(f"Inference YOLO done ✅ .")

        return



    # def _prepare_inference(self, yolo_weights:str = None) -> str:
    #     """ Prepares inference with YOLO. """

    #     # get model:
    #     os.chdir(self.yolov5dir)
    #     if yolo_weights is None:
    #         raise NotImplementedError()
    #         weights_dir = utils_yolo.get_last_weights()
    #     else:
    #         weights_dir = os.path.dirname(yolo_weights)

    #     self.log.info(f"Prepared YOLO for inference ✅ .")

    #     return weights_dir



    






    # def _edit_yaml(self) -> None:
    #     """ Edits YAML data file from yolov5. """

    #     self.log.info("⏳ Setting configurations for YOLO: ")
    #     classes = dict([(value, key) for key, value in self.map_classes.items()])
       
    #     # raise Exception
    #     yaml_fp = os.path.join(self.yolov5dir, 'data', 'helical.yaml')
    #     text = {'path':self.data_folder, 'train': os.path.join(self.data_folder, 'train', 'images'), 'val': os.path.join(self.data_folder, 'val', 'images'), 'test': os.path.join(self.data_folder, 'test', 'images'), 'names':classes}
    #     with open(yaml_fp, 'w') as f:
    #         yaml.dump(data = text, stream=f)
    #     self.log.info(f"✅ YOLO set up completed YOLO ✅ .")

    #     return yaml_fp
    
    # def save_training_data(self, weights:str, start_time:str) -> None:
        # """ Saves training data into a json file in the runs folder from YOLO. """

        # # get file splitting: 
        # if os.path.isdir(self.data_folder.replace('tiles', 'wsi')):
        #     sets = ['train', 'val', 'test']
        #     data = {}
        #     for dirname in sets: 
        #         dirpath = os.path.join(self.data_folder, dirname, 'images')
        #         data[dirname] = [file for file in os.listdir(dirpath) if 'DS' not in file ]
        # print(f"dictionary data: {data}")
        # # get training duration:
        # end_time = time.time()
        # train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        # # save info into json file:
        # otherinfo_yolo = {'datafolder': self.data_folder, 'data':data, 'classes': self.map_classes, 'epochs': self.epochs, 'duration': train_yolo_duration, 'weights': {weights}}
        # utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = os.path.join(self.yolov5dir, 'runs', 'train'))
        # self.log.info(f"Training YOLO done ✅ . Training duration: {train_yolo_duration}")

        # return






def test_YOLODetector(): 

    system = 'mac'
    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else 'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else 'C:\marco\yolov5'
    data_folder = '/Users/marco/Downloads/try_train/detection/tiles' if system == 'mac' else r'C:\marco\biopsies\muw\detection\tiles'
    map_classes = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3}
    tile_size = 512
    batch_size=8
    epochs=1
    conf_thres=0.7
    images_dir = '/Users/marco/Downloads/test_folders/test_featureextractor/images'
    weights = '/Users/marco/yolov5_copy/runs/train/exp4/weights/best.pt'
    augment = False
    conf_thres = 0.8




    detector = YOLODetector(images_dir = images_dir,
                            weights = weights,
                            yolov5dir=yolov5dir,
                            repository_dir=repository_dir,
                            # map_classes: dict = {'Glo-healthy':0, 'Glo-NA':1, 'Glo-unhealthy':2, 'Tissue':3},
                            # system = 'mac',
                            augment=augment,
                            # tile_size = 512,
                            # batch_size = 8,
                            # epochs = 3,
                            conf_thres = conf_thres)
    detector.infere()

    return
        


if __name__ == '__main__':
    
    test_YOLODetector()
