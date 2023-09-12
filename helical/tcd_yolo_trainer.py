import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import Literal, List
import shutil
# from plotter_data_detect_muw_sfog import PlotterDetectMUW
# from plotter_data_detect_hubmap_pas import PlotterDetectHub
# from plotter_data_segm_hubmap_pas import PlotterSegmHubmap
from tcd_yolo_base import YOLOBase
from exp_tracker import Exp_Tracker
from datetime import datetime
from time import time
from utils import get_config_params


class YOLO_Trainer(YOLOBase):

    def __init__(self, 
                config_yaml_fp:str,
                ) -> None: 
        
        super().__init__(config_yaml_fp)
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='yolo_trainer')
        self._class_name = self.__class__.__name__
        self.task = self.params["task"]
        def_weights = 'yolov5s-seg.pt' if self.task == 'segmentation' else 'yolov5s.pt' 
        self.weights = def_weights if self.params["yolov5_weights"] is None else self.params["yolov5_weights"]
        self.exp_fold = os.path.basename(self.get_exp_fold(os.path.join(self.params['yolov5dir'], 'runs', 'train')))
        now = datetime.now()
        self.exp_datetime = now.strftime("%d/%m/%Y %H:%M:%S")
        other_params = {'datetime': self.exp_datetime, 'duration':'0h 0m 0s', 'image_size':self.params['image_size'], 'task':self.params['task'], 
                        'exp_fold':self.exp_fold, 'weights':self.params['yolov5_weights'], 'crossvalidation':self.crossvalidation, 
                        'tot_kfolds': self.params['crossvalid_tot_kfolds'], 'cur_kfold': self.params['crossvalid_cur_kfold'], 'status':'started', 'note':self.params['note']}
        self.tracker = Exp_Tracker(other_params=other_params)
        # self.tracker.update_tracker(**kwargs)

        return
    

    def train(self) -> None:
        """   Runs the YOLO detection model. """
        class_name = self.__class__.__name__
        
        # 1) prepare training:
        if self.crossvalidation: 
            self.crossvalidator._change_kfold(self.cur_kfold)
        # start_time = time.time()
        yaml_fn = self._edit_yaml()
        self.log.info(f"{class_name}.{'train'}: data:{yaml_fn}")
        self.log.info(f"{class_name}.{'train'}: weights:{self.weights}")

        # 2) train:
        try:
            # train_start = time()
            duration_start = datetime.now()
            self.tracker.update_status('prepared')
            self.log.info(f"⏳ Start training YOLO:")
            os.chdir(self.yolov5dir)
            script = "segment/train.py" if self.task == 'segmentation' else 'train.py'
            prompt = f"python {script} --img {self.params['image_size']} --batch {self.batch_size} --epochs {self.epochs}"
            prompt += f" --data {yaml_fn} --weights {self.weights} --workers {self.workers}"
            prompt = prompt+f" --device {self.device}" if self.device is not None else prompt 
            prompt = prompt+f" --single-cls" if self.single_cls is True else prompt 
            self.log.info(f"{class_name}.{'train'}: {prompt}")
            os.system(prompt)
            os.chdir(self.repository_dir)
        except:
            self.tracker.update_status('train_err')

        # 3) save:
        train_h, train_m, train_s = self._get_train_duration(start=duration_start)
        train_duration = f"{train_h}h {train_m}m {train_s}s"
        print(f"other train dur: {train_duration}")
        if train_m < 1: 
            print('a')
            self.tracker.update_duration(train_duration)
            self.tracker.update_status('<1min')
            return

        try:
            print('b')
            self.tracker.update_duration(train_duration)
            self._log_data(mode='train')
            self.tracker.update_status('completed')
        except:
            print('c')
            self.tracker.update_duration(train_duration)
            self.tracker.update_status('plotfailed')
    
        return


    def _log_data(self, mode = Literal['train', 'detect']): 
        """ Logs a bunch of dataset info prior to training. """

        exp_fold = os.path.join(self.yolov5dir, 'runs', f"{mode}", self.exp_fold)
        print(f"exp fold for plotting: {exp_fold}")
        shutil.copyfile(src=os.path.join(self.repository_dir, 'code.log'), dst = os.path.join(exp_fold, 'train.log'))
        shutil.copyfile(src=os.path.join(self.repository_dir, 'exp_tracker.csv'), dst = os.path.join(exp_fold, 'exp_tracker.csv'))
        data_root = os.path.join(self.data_folder.split(self.task)[0], self.task)
        # plotters = {'detection':{'muw': PlotterDetectMUW, 
        #                         'hubmap': PlotterDetectHub},
        #             'segmentation':{'muw':None, 
        #                             'hubmap':PlotterSegmHubmap}}

        # plotter = plotters[self.task][self.dataset]
        # assert plotter is not None, self.log.error(f"{self._class_name}.{'_log_data'}: Plotter for segmentation muw Not implemented")
                
        # try:
        #     plotter(data_root=data_root, files=None, verbose = False)
        #     shutil.copyfile(src=os.path.join(self.repository_dir, 'plot_data.png'), dst = os.path.join(exp_fold, 'data.png'))

        # except:
        #     self.log.error(f"{self._class_name}.{'_log_data'}: ❌ Failed plotting.")

        return






# def test_YOLODetector(): 

#     import sys 
#     system = 'mac' if sys.platform == 'darwin' else 'windows'

#     repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
#     yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
#     data_folder = '/Users/marco/helical_tests/test_hubmap_manager/detection/tiles' if system == 'mac' else r'D:\marco\datasets\slides\detection\tiles'
#     device = None if system == 'mac' else 'cuda:0'
#     workers = 0 if system == 'mac' else 1
#     map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} #{'glomerulus':0}  
#     save_features = False
#     tile_size = 512 
#     batch_size=2 if system == 'mac' else 4
#     epochs=1   
#     dataset = 'hubmap'
      
#     tracker = update_tracker(dataset = dataset,
#                             data_folder=data_folder, 
#                             repository_dir=repository_dir,
#                             yolov5dir=yolov5dir,
#                             map_classes=map_classes,
#                             tile_size = tile_size,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             workers=workers,
#                             device=device,
#                             save_features=save_features)

#     detector = YOLODetector(dataset = dataset,
#                             data_folder=data_folder, 
#                             repository_dir=repository_dir,
#                             yolov5dir=yolov5dir,
#                             map_classes=map_classes,
#                             tile_size = tile_size,
#                             batch_size=batch_size,
#                             epochs=epochs,
#                             workers=workers,
#                             device=device,
#                             save_features=save_features)
#     detector.train()

#     return
        

def test_yolo_detector(): 
    config_yaml_fp = '/Users/marco/yolo/code/helical/tcd_config_training.yaml'
    detector = YOLO_Trainer(config_yaml_fp=config_yaml_fp)
    detector.train()

    return

if __name__ == '__main__':
    
    test_yolo_detector()
