
import os, shutil
import time
from typing import Any
from tqdm import tqdm
import torch
from datetime import datetime
from torch import nn
import seaborn as sns
from glob import glob
from sklearn.metrics import confusion_matrix
from utils import get_config_params
from cnn_trainer_base import CNN_Trainer_Base
import matplotlib.pyplot as plt



class CNN_Inferer(CNN_Trainer_Base):

    def __init__(self, config_yaml_fp: str) -> None:
        super().__init__(config_yaml_fp)
        self.params = get_config_params(config_yaml_fp,'cnn_inferer')
        self._set_new_attrs()
        os.makedirs(self.save_dir)
        return
    
    def _set_new_attrs(self)->None:
        self.model_path = self.params['model_path'] if self.params['model_path'] is not None else self._get_last_model_path()
        save_dir = self.params['save_dir']
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = os.path.join(save_dir, 'infer', f"{dt_string}")
        self.all_true_classes = []
        self.all_preds = []
        return
    

    def _get_last_model_path(self)->str:
        """ Gets last (modified on MacOS/created on Windows) fold."""
        func_n = self._get_last_model_path.__name__
        folds = [os.path.join(self.cnn_exp_fold, fold) for fold in os.listdir(self.cnn_exp_fold) if "DS" not in fold]
        last_exp = max(folds, key=os.path.getctime)
        assert os.path.isdir(last_exp), self.assert_log(f"CNN experiment last fold: {last_exp} is not a valid dirpath.", func_n=func_n)
        model_path = glob(os.path.join(last_exp, '*.pt'))
        assert len(folds)>0, self.assert_log(f"CNN experiment dir: {self.cnn_exp_fold} looks empty!", func_n=func_n)
        assert len(model_path)==1, self.assert_log(f"Found {len(model_path)} models within last CNN exp dir:{last_exp}", func_n=func_n)
        return model_path[0]  
    

    def _save_inferred(self, preds:torch.Tensor, fps:list)->None:
        """ Saves prediction in self.save_dir fold."""
        
        preds = preds.cpu().detach().numpy()
        reversed_map_classes = {v:k for k,v in self.map_classes.items()}


        for pred, src in zip(preds, fps):
            pred = int(pred)
            lbl_name = reversed_map_classes[pred]
            dst = os.path.join(self.save_dir, lbl_name, os.path.basename(src))
            shutil.copy(src, dst)


        return


    def eval_model(self)->None:
        func_n = self.eval_model.__name__

        # check there is at least 1 param requiring grad
        model = self.get_model()
        model.load_state_dict(torch.load(self.model_path))  # load trained model
        if self.device != 'cpu': model.to(self.device)   # move model to gpus
        assert any([param.requires_grad for param in model.features.parameters()]), self.assert_log(f"No param requires grad. Model won't update.", func_n=func_n)

        # make output folders
        for clss in self.map_classes.keys():
            os.makedirs(os.path.join(self.save_dir, clss))

        # test_batches = len(self.dataloaders['val'])
        print("Evaluating model")
        print('-' * 10)
        assert len(self.loaders['test'])>0
        model.train(False)
        model.eval()
        since = time.time()
        for data in tqdm(self.loaders['test'], desc='Inferring on test set'):
            inputs, fps = data
            # move data to gpu
            if self.device != 'cpu': inputs= inputs.to(self.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1, keepdim=True)
            self._save_inferred(preds=preds, fps=fps)
            del inputs, outputs, preds
            torch.cuda.empty_cache()

        # print duration
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

        return
    
    def __call__(self) -> Any:
        self.get_loaders(mode='inference')
        self._get_last_model_path()
        self.eval_model()

        return 