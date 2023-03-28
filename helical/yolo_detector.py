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
from yolo_base import YOLOBase
import shutil
from profiler_hubmap import ProfilerHubmap
from plotter_data_hubmap import Plotter

class YOLODetectorTrainer(YOLOBase):

    def __init__(self, 
                 *args, 
                 **kwargs
                ) -> None: 

        super().__init__(*args, **kwargs)
        self._class_name = self.__class__.__name__
        self.task = 'detection'
        self.def_weights = 'yolov5s.pt' 

        return
    

    def train(self, weights: str = None) -> None:
        """   Runs the YOLO detection model. """
        class_name = self.__class__.__name__
        
        # 1) prepare training:
        # start_time = time.time()
        yaml_fn = os.path.basename(self._edit_yaml())
        weights = weights if weights is not None else self.def_weights
        self.log.info(f"{class_name}.{'train'}: weights:{weights}")

        # 2) train:
        self.log.info(f"⏳ Start training YOLO:")
        os.chdir(self.yolov5dir)
        prompt = f"python train.py --img {self.tile_size} --batch {self.batch_size} --epochs {self.epochs}"
        prompt += f" --data {yaml_fn} --weights {weights} --workers {self.workers}"
        prompt = prompt+f" --device {self.device}" if self.device is not None else prompt 
        self.log.info(f"{class_name}.{'train'}: {prompt}")
        os.system(prompt)
        os.chdir(self.repository_dir)

        # 3) save:
        self._log_data(mode='train')
    
        return


    def _log_data(self, mode = Literal['train', 'detect']): 
        """ Logs a bunch of dataset info prior to training. """

        exp_root = os.path.join(self.yolov5dir, 'runs', f"{mode}")
        exp_fold = self.get_exp_fold(exp_fold=exp_root)
        shutil.copyfile(src=os.path.join(self.repository_dir, 'code.log'), dst = os.path.join(exp_fold, 'train.log'))
        data_root = os.path.join(self.data_folder.split(self.task)[0], self.task)

        if self.dataset == 'hubmap':
            try:
                profiler = ProfilerHubmap(data_root=data_root, 
                                        wsi_images_like = self.wsi_images_like, 
                                        wsi_labels_like = self.wsi_labels_like,
                                        tile_images_like = self.tile_images_like,
                                        tile_labels_like = self.tile_labels_like,
                                        n_classes=self.n_classes)
                profiler()
                shutil.copyfile(src=os.path.join(self.repository_dir, 'code.log'), dst = os.path.join(exp_fold, 'data_summary.log'))
            except:
                self.log.error(f"{self._class_name}.{'_log_data'}: ❌ Failed data profiling.")
            try:
                plotter = Plotter(data_root=data_root, 
                                files=None, 
                                verbose = False,
                                wsi_images_like = self.wsi_images_like, 
                                wsi_labels_like = self.wsi_labels_like,
                                tile_images_like = self.tile_images_like,
                                tile_labels_like = self.tile_labels_like,
                                n_classes=self.n_classes,
                                empty_ok=False) 
                plotter()
                shutil.copyfile(src=os.path.join(self.repository_dir, 'plot_data.png'), dst = os.path.join(exp_fold, 'data.png'))
            except:
                self.log.error(f"{self._class_name}.{'_log_data'}: ❌ Failed plotting.")

        else:
            raise NotImplementedError()


        return




def runner():

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_hubmap_manager/detection/tiles' if system == 'mac' else r'D:\marco\datasets\slides\detection\tiles'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} #{'glomerulus':0}  
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=1   
    dataset = 'hubmap' 
    detector = YOLODetectorTrainer(dataset = dataset,
                                    data_folder=data_folder, 
                                    repository_dir=repository_dir,
                                    yolov5dir=yolov5dir,
                                    map_classes=map_classes,
                                    tile_size = tile_size,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    workers=workers,
                                    device=device,
                                    save_features=save_features)
    detector.train()


    return


if __name__ == '__main__':
    
    runner()
