import torch
from torchvision import models
from torch import nn
from cnn_trainer_base import CNN_Trainer_Base
from cnn_loaders import CNNDataLoaders  
import os, sys
from cnn_crossvalidation import CNN_KCrossValidation
from utils import get_config_params 
from loggers import get_logger


class CNN_Trainer(CNN_Trainer_Base):

    def __init__(self, config_yaml_fp:str) -> None:

        super().__init__(config_yaml_fp)
        self.class_name = self.__class__.__name__
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(config_yaml_fp, 'cnn_trainer')
        return
    

    def _crossvalidation(self)->None:
        """ Applies Crossvalidation to train, val, test sets. Applies a total of 'k_tot' 
            crossvalidation folds and uses as current test fold 'k_i' fold."""
        func_n = self._crossvalidation.__name__
        self.format_msg(msg=f"⏳ Applyting crossvalidation: tot:{self.k_tot}, current k_fold:{self.k_i}", func_n=func_n)
        crossvalidator = CNN_KCrossValidation(data_root=os.path.join(self.cnn_data_fold, 'cnn_dataset'), k=self.k_tot, dataset = self.dataset, )
        crossvalidator._change_kfold(fold_i=self.k_i)
        self.format_msg(msg=f"✅ Applied crossvalidation. Current k fold: {self.k_i}:", func_n=func_n)
        return
    
    
    def get_loaders(self)->None:
        """ Creates DataLoader class and gets Train and Val Loaders. """
        func_n = self.get_loaders.__name__
        self.format_msg(f"⏳ Getting loaders", func_n=func_n)
        self.dataloader_cls = CNNDataLoaders(root_dir=os.path.join(self.cnn_data_fold, 'cnn_dataset'), map_classes=self.map_classes, 
                                        batch=self.batch, num_workers=self.num_workers)
        self.loaders = self.dataloader_cls()
        self.format_msg(f"✅ Got loaders.", func_n=func_n)
        return
    
    

    

    def train(self)-> None:
        """ Trains the CNN VGG16 model. """
        func_n = self.train.__name__
        self.format_msg(f"⏳ Training VGG16 model.", func_n=func_n)

        vgg16 = self.train_model()
        os.makedirs(self.weights_save_fold)
        torch.save(vgg16.state_dict(), os.path.join(self.weights_save_fold, 'VGG16_v2-OCT_Retina_half_dataset.pt'))
        self.format_msg(f"✅ Trained VGG16 model. Best model saved in {self.weights_save_fold}", func_n=func_n)
        # eval_model(model=vgg16, dataloader_cls=self.dataloader_cls, dataloaders=self.loaders, criterion=criterion)
        return


    def __call__(self)->None:

        self._crossvalidation()
        self.get_loaders()
        self.train()

        return



if __name__ == '__main__':
    config_yaml_fp = '/Users/marco/yolo/code/helical/config_tcd.yaml'
    cnn_trainer = CNN_Trainer(config_yaml_fp=config_yaml_fp)
    cnn_trainer()

