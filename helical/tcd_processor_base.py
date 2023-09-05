
from typing import Literal, Tuple
import os
from splitter import Splitter
from move_data import move_slides_for_tiling, move_slides_back_from_tiling
from cleaner_muw import CleanerMuw
from configurator import Configurator
from abc import ABC, abstractmethod
from utils import get_config_params
from loggers import get_logger
import numpy as np


class ProcessorBase(Configurator, ABC): 
    def __init__(self,
                 config_yaml_fp:str
                 ) -> None:
        
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='processor')
        self.log = get_logger()
        super().__init__()
        self.class_name = self.__class__.__name__
        self._set_all_attrs()
        self._parse_args()

        return
    

    def _set_all_attrs(self)->None:
        """ Sets all class attributes. """
        self.src_root = self.params["src_root"]
        self.dst_root = self.params["dst_root"]
        self.slide_format = self.params["slide_format"]
        self.label_format = self.params["label_format"]
        self.tiling_shape = self.params["tiling_shape"]
        self.tiling_step = self.params["tiling_step"]
        self.tiling_level = self.params["tiling_level"]
        self.tiling_show = self.params["tiling_show"]
        self.split_ratio = self.params["split_ratio"]
        self.remove_classes = self.params['remove_classes']
        self.ignore_classes = self.params['ignore_classes']
        self.task = self.params["task"]
        self.multiple_samples = self.params['multiple_samples']
        self.verbose = self.params["verbose"]
        self.safe_copy = self.params["safe_copy"]
        self.reproducibility = self.params["reproducibility"]
        self.data_source = self.params['datasource']
        self.wsi_dir = os.path.join(self.dst_root, self.task, 'wsi')
        self.tiles_dir = os.path.join(self.dst_root, self.task, 'tiles')
        self.verbose_level = self.params['verbose_level']
        self.map_classes = self.params['map_classes']
        self.stain = self.params['stain']
        self.resize = self.params['resize']

        return
    
    
    def _parse_args(self):
        func_n = self._parse_args.__name__

        allowed = get_config_params('config_allowed.yaml', 'allowed')
        assert not (self.ignore_classes is not None and self.remove_classes is not None), self.assert_log(f"'ignore_classes' and 'remove_classes' can't be both not None at same time.",func_n=func_n)
        assert os.path.isdir(self.params['src_root']), self.assert_log( f"Repo:{self.params['repo']} is not a valid dirpath.",func_n=func_n)
        assert os.path.isdir(self.params['dst_root']), self.assert_log( f"Repo:{self.params['repo']} is not a valid dirpath.",func_n=func_n)
        assert isinstance(self.map_classes, dict), self.assert_log(f"Map_classes:{self.map_classes} should be a dict.",func_n=func_n)
        assert 0 in self.map_classes.values(), self.assert_log(f"Map_classes:{self.map_classes} should be zero indexed.",func_n=func_n)
        assert self.params['slide_format'] in allowed['slide_formats'], self.assert_log(f"'slide_format':{self.params['slide_format']} should be one of {allowed['slide_formats']}",func_n=func_n)
        assert self.params['label_format'] in allowed['label_formats'], self.assert_log(f"'label_format':{self.params['label_format']} should be one of {allowed['label_formats']}",func_n=func_n)
        assert self.params['datasource'] in allowed['datasources'], self.assert_log(f"'datasource':{self.params['datasource']} should be one of {allowed['datasources']}",func_n=func_n)
        assert self.params['task'] in allowed['tasks'], self.assert_log(f"'task':{self.params['task']} should be one of {allowed['tasks']}",func_n=func_n)
        assert self.params['stain'] in allowed['stains'], self.assert_log(f"'stain':{self.params['stain']} should be one of {allowed['stains']}",func_n=func_n)
        assert isinstance(self.params['split_ratio'], list), self.assert_log(f"'split_ratio':{self.params['split_ratio']} should be type list, but is {type(self.params['split_ratio'])}",func_n=func_n)
        assert (len(self.params['split_ratio']) == 3 or len(self.params['split_ratio']) == 2) and round(np.sum(np.array(self.params['split_ratio'])), 2) == 1.0, self.assert_log(f"'split_ratio' should be a list of floats with sum 1, but has sum {np.sum(np.array(self.params['split_ratio']))}.",func_n=func_n )
        assert isinstance(self.params['tiling_shape'], list), self.assert_log(f"'tiling_shape':{self.params['tiling_shape']} should be type list, but is {type(self.params['tiling_shape'])}",func_n=func_n)
        assert len(self.params['tiling_shape']) == 2, self.assert_log(f"'tiling_shape':{self.params['tiling_shape']} should have length = 2, but list, but has length{len(self.params['tiling_shape'])}",func_n=func_n)
        assert isinstance(self.params['safe_copy'], bool), self.assert_log(f"'safe_copy':{self.params['safe_copy']} should be boolean but is {type(self.params['safe_copy'])}",func_n=func_n)
        assert isinstance(self.params['tiling_level'], int), self.assert_log(f"'tiling_level':{self.params['tiling_level']} should be int, but is type {type(self.params['tiling_level'])}",func_n=func_n)
        if self.params['resize'] is not None: assert isinstance(self.params['resize'], list), self.assert_log(f"'resize':{self.params['resize']} should be type list, but is {type(self.params['resize'])}",func_n=func_n)
        if self.params['resize'] is not None: assert len(self.params['resize']) == 2, self.assert_log(f"'resize':{self.params['resize']} should have length = 2, but list, but has length{len(self.params['resize'])}",func_n=func_n)
        
        return
        
        
    def _make_tiles_branch(self) -> None: 
        """ Creates a tree for tiles with same structures as the one for wsi: tiles -> train,val,test -> images,labels"""
        
        # 1) makedirs:
        if self.verbose_level == 'high': self.log.info(f"{self.__class__.__name__}.{'_make_tiles_branch'}: Creating new tiles branch:")
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(self.tiles_dir, subfold, subsubfold), exist_ok=True)
        self.log.info(f"{self.__class__.__name__}.{'_make_tiles_branch'}: Created tree at '{self.tiles_dir}'.")

        return  
    

    def _split_data(self) -> None:
        """ Splits WSIs based on 'split_ratio' attribute. """

        if self.verbose is True: 
            self.log.info(f"{self.class_name}.{'_split_data'}: ⏳ Splitting WSIs with {self.params['ratio']} ratio.: ")
        splitter = Splitter(config_yaml_fp=self.config_yaml_fp)

        splitter()
        self.log.info(f"{self.class_name}.{'_split_data'}: ✅ Splitted WSIs in train, val, test sets.")
        
        return
    

    def _move_slides_forth(self) -> None: 
        """ Moves slides together with labels to be ready for tiling. """

        move_slides_for_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        if self.verbose_level == 'high': self.log.info(f"{self.__class__.__name__}.{'_move_slides_forth'}: Moved slides forth.")

        return
    
    
    def _move_slides_back(self) -> None:
        """ Moves slides together with labels to be ready for tiling. """

        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_back'}: Moving slides back:")
        move_slides_back_from_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_back'}: Moved slides back.")

        return
    
    
    def _clean_balance_dataset(self, safe_copy:bool=False) -> None: 
        """ Uses the dataset cleaner to finalize the dataset, e.g. by grouping classes 
            from {0:glom_healthy, 1:glom_na, 2: glom_sclerosed, 3: tissue}
            to {0:glom_healthy, 1:glom_sclerosed} """
        
        cleaner = CleanerMuw(data_root=os.path.join(self.dst_root, self.task), 
                             safe_copy=safe_copy,
                             remove_classes=self.remove_classes,
                             ignore_classes=self.ignore_classes)
        if self.data_source == 'muw':
            cleaner._clean_muw()
        else:
             cleaner._clean_generic()

        return
    

    def tile_dataset(self): 
        """ Tiles the whole dataset: train, val, test. """

        self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: ⏳ Tiling dataset at '{self.wsi_dir}':")
        datasets = ['train', 'val', 'test']
        for dataset in datasets: 
            self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: START TILING '{dataset.upper()}'")
            self._tile_folder(dataset=dataset)
        self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: ✅ Tiled '{self.wsi_dir}'.")

        return
    
    @abstractmethod
    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        return
    
    
    def _rename_tiff2tif(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.tiff' in file]
        old_new_names = [(file, file.replace('.tiff', '.tif')) for file in files ]
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return
    

    

    # def __call__(self) -> None:

    #     # 1) create tiles branch
    #     self._make_tiles_branch()
    #     # 1) split data
    #     self._split_data()
    #     # 2) prepare for tiling 
    #     self._move_slides_forth()
    #     # 3) tile images and labels:
    #     self.tile_dataset()
    #     #4) move slides back 
    #     self._move_slides_back()
    #     # 4) clean dataset, e.g. 
    #     self._clean_muw_dataset()

    #     return




# if __name__ == '__main__': 
#     test_ProcessorManager()