
import os, shutil
import time
from typing import Any
from tqdm import tqdm
import torch
from torch import nn
from glob import glob
import numpy as np

from utils import get_config_params
from cnn_trainer_base import CNN_Trainer_Base



class CNN_FeatureExtractor(CNN_Trainer_Base):

    def __init__(self, config_yaml_fp: str) -> None:
        super().__init__(config_yaml_fp)
        self.params = get_config_params(config_yaml_fp,'cnn_feature_extractor')
        self._set_new_attrs()
        return
    
    def _set_new_attrs(self)->None:
        self.model_path = self.params['model_path'] if self.params['model_path'] is not None else self._get_last_model_path()
        self.save_dir = self.params['save_dir']
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
    
    def _create_dirs(self)->None:
        # 1) create dirs:
        if os.path.isdir(self.save_dir): shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)     
        return
    
    
    def save_features(self, outputs:torch.Tensor, fps:list)->None:
        """ Saves extracted features into save_dir. """
        outputs = outputs.cpu().detach().numpy()
        for feats, fp in zip(outputs, fps):
            save_fp = os.path.join(self.save_dir, os.path.basename(fp).split('.')[0])
            np.save(save_fp, feats)
        return


    def extract_features(self)->None:
        func_n = self.extract_features.__name__

        # check there is at least 1 param requiring grad
        model = self.get_model()
        model.load_state_dict(torch.load(self.model_path))  # load trained model
        model.classifier = model.classifier[:-1] # remove last layer
        if self.device != 'cpu': model.to(self.device)   # move model to gpus
        assert any([param.requires_grad for param in model.features.parameters()]), self.assert_log(f"No param requires grad. Model won't update.", func_n=func_n)
        since = time.time()

        # test_batches = len(self.dataloaders['val'])
        print("Evaluating model")
        print('-' * 10)

        for _set in ['train', 'val', 'test']:
            progress_bar = tqdm(self.loaders['val'], desc = f"Extracting feats in {_set}")
            for i, data in enumerate(progress_bar):
                model.train(False)
                model.eval()
                inputs, _, fps = data
                # move data to gpu
                if self.device != 'cpu': inputs= inputs.to(self.device)
                outputs = model(inputs)
                self.save_features(outputs=outputs, fps=fps)
                if i==0: progress_bar.set_description(desc=f"Extracting feats in {_set} with shape:{list(outputs.shape)}")


        elapsed_time = time.time() - since
        print()
        print("Feature extraction completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

        return
    
    def __call__(self) -> Any:
        self._create_dirs()
        self.get_loaders(return_fp=True, show_data=False)
        self._get_last_model_path()
        self.extract_features()

        return 