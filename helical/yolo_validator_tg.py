import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from loggers import get_logger
import torch
from utils import get_config_params


class YOLO_Validator():

    def __init__(self, 
                 yaml_fp:str,
                 trained_model_weights: str,
                 conf_thres: float,
                 ) -> None: 

        self.log = get_logger()
        self.params = get_config_params(yaml_fp=yaml_fp, config_name='validation')
        self.trained_model_weights = trained_model_weights
        self.conf_thres = conf_thres
        self.yaml_path = os.path.join(self.params['yolov5dir'], 'data', 'tg_data.yaml')

        return
    
    # def _parse_args(self):

    #     assert os.path.isdir(self.params['images_dir']), ValueError(f"'images_dir': {self.params['images_dir']} is not a valid dirpath.")
    #     assert os.path.isdir(self.params['yolov5dir']), ValueError(f"'yolov5dir': {self.params['yolov5dir']} is not a valid dirpath.")
    #     return


    def validate(self) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        os.chdir(self.params['yolov5dir'])
        if self.params['device'] == 'gpu':
            device='cuda:0' if torch.cuda.is_available() else 'cpu'

        # 2) define command:
        command = f"python val.py --data {self.yaml_path} --weights {self.trained_model_weights} --device {device}"
        if self.params['augment'] is True:
            command += " --augment"
        if self.params['save_txt'] is True: 
           command +=" --save-txt"
        if self.conf_thres is not None:
            command += f" --conf-thres {self.conf_thres}" 
        if self.params['output_dir'] is not None:
            command += f" --project {os.path.split(self.params['output_dir'])[0]}" 
        if self.params['output_dir'] is not None:
            command += f" --name {os.path.split(self.params['output_dir'])[1]}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.params['repository_dir'])
        self.log.info(f"Inference YOLO done ✅ .")

        return





