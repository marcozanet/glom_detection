
from typing import Literal, Tuple
import os
from loggers import get_logger
from decorators import log_start_finish
from converter_muw import ConverterMuW
from converter_hubmap import ConverterHubmap
from splitter import Splitter
from move_data import move_slides_for_tiling, move_slides_back_from_tiling
from tiling import Tiler
from tiler_hubmap import TilerHubmap
import shutil
from cleaner_muw import CleanerMuw
from configurator import Configurator
from abc import ABC, abstractmethod



class ManagerBase(Configurator, ABC): 
    def __init__(self,
                data_source: Literal['muw', 'hubmap'],
                src_root: str, 
                dst_root: str, 
                stain: Literal['sfog', 'pas'],
                slide_format: Literal['tif', 'tiff'],
                label_format: Literal['gson', 'mrxs.gson'],
                tiling_shape: Tuple[int],
                tiling_step: int,
                tiling_level: int,
                tiling_show: bool = True,
                split_ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                safe_copy: bool = False,
                verbose: bool = False,
                empty_perc: float = 0.1,
                reproducibility: bool = True) -> None:
        
        super().__init__()
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.slide_format = slide_format
        self.label_format = label_format
        self.tiling_shape = tiling_shape
        self.tiling_step = tiling_step
        self.tiling_level = tiling_level
        self.tiling_show = tiling_show
        self.split_ratio = split_ratio
        self.stain = stain
        self.task = task
        self.verbose = verbose
        self.safe_copy = safe_copy
        self.reproducibility = reproducibility
        self.wsi_dir = os.path.join(dst_root, self.task, 'wsi')
        self.tiles_dir = os.path.join(dst_root, self.task, 'tiles')
        self.data_source = data_source
        self.class_name = self.__class__.__name__
        
        return


    def _make_tiles_branch(self) -> None: 
        """ Creates a tree for tiles with same structures as the one for wsi: tiles -> train,val,test -> images,labels"""
        
        # 1) makedirs:
        self.log.info(f"{self.__class__.__name__}.{'_make_tiles_branch'}: Creating new tiles branch:")
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(self.tiles_dir, subfold, subsubfold), exist_ok=True)
        self.log.info(f"{self.__class__.__name__}.{'_make_tiles_branch'}: Created new tiles branch at '{self.tiles_dir}'.")

        return  
    

    def _split_data(self) -> None:
        """ Splits WSIs based on 'split_ratio' attribute. """

        if self.verbose is True: 
            self.log.info(f"{self.class_name}.{'_split_data'}: splitting WSIs: ")
        splitter = Splitter(src_dir=self.src_root,
                            dst_dir=self.dst_root,
                            image_format=self.slide_format,
                            ratio=self.split_ratio,
                            task=self.task,
                            verbose = self.verbose, 
                            safe_copy = self.safe_copy,
                            reproducibility=self.reproducibility)
        splitter()
        self.log.info(f"{self.class_name}.{'_split_data'}: splitted WSIs with {self.split_ratio} ratio.")
        
        return
    

    def _move_slides_forth(self) -> None: 
        """ Moves slides together with labels to be ready for tiling. """

        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_forth'}: Moving slides forth:")
        move_slides_for_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_forth'}: Moved slides forth.")

        return
    
    
    def _move_slides_back(self) -> None:
        """ Moves slides together with labels to be ready for tiling. """

        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_back'}: Moving slides back:")
        move_slides_back_from_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        if self.verbose is True:
            self.log.info(f"{self.__class__.__name__}.{'_move_slides_back'}: Moved slides back.")

        return
    
    
    def _clean_muw_dataset(self, safe_copy:bool=False) -> None: 
        """ Uses the dataset cleaner to finalize the dataset, e.g. by grouping classes 
            from {0:glom_healthy, 1:glom_na, 2: glom_sclerosed, 3: tissue}
            to {0:glom_healthy, 1:glom_sclerosed} """
        
        cleaner = CleanerMuw(data_root=os.path.join(self.dst_root, self.task), safe_copy=safe_copy)
        cleaner._clean_muw()
        

        return
    

    def tile_dataset(self): 
        """ Tiles the whole dataset: train, val, test. """

        self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: ⏳ Tiling dataset at '{self.wsi_dir}':")
        datasets = ['train', 'val', 'test']
        for dataset in datasets: 
            self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: Tiling '{dataset}'")
            self._tile_folder(dataset=dataset)
        self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: ✅ Tiled '{self.wsi_dir}'.")

        return
    
    @abstractmethod
    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        # """ Tiles a single folder"""
        # class_name = self.__class__.__name__
        # func_name = '_tile_folder'

        # slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        # save_folder_labels = os.path.join(self.tiles_dir, dataset)
        # save_folder_images = os.path.join(self.tiles_dir, dataset)

        # # 1) convert annotations to yolo format:
        # self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ⏳    ########################")
        # if self.data_source == 'muw':
        #     converter = ConverterMuW(folder = slides_labels_folder, 
        #                             convert_from='gson_wsi_mask',  
        #                             convert_to='txt_wsi_bboxes',
        #                             save_folder= slides_labels_folder, 
        #                             level = self.tiling_level,
        #                             verbose=self.verbose)
        # elif self.data_source == 'hubmap':
        #     converter = ConverterHubmap(folder = slides_labels_folder, 
        #                                 convert_from='json_wsi_mask',  
        #                                 convert_to='txt_wsi_bboxes',
        #                                 save_folder= slides_labels_folder, 
        #                                 level = self.tiling_level,
        #                                 verbose=self.verbose)
        # converter()
        # self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ✅    ########################")


        # # 2) tile images:
        # self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        # if self.data_source == 'muw':
        #     tiler = Tiler(folder = slides_labels_folder, 
        #                 tile_shape= self.tiling_shape, 
        #                 step=self.tiling_step, 
        #                 save_root= save_folder_images, 
        #                 level = self.tiling_level,
        #                 show = self.tiling_show,
        #                 verbose = self.verbose)
        # elif self.data_source == 'hubmap': 
        #     print(f"alling tiler hubmap")
        #     tiler = TilerHubmap(folder = slides_labels_folder, 
        #                         tile_shape= self.tiling_shape, 
        #                         step=self.tiling_step, 
        #                         save_root= save_folder_images, 
        #                         level = self.tiling_level,
        #                         show = self.tiling_show,
        #                         verbose = self.verbose)
        # target_format = 'tif'
        # tiler(target_format=target_format)
        # self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

        # # 3) tile labels:
        # self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ⏳    ########################")
        # target_format = 'txt'
        # tiler = Tiler(folder = slides_labels_folder, 
        #             tile_shape= self.tiling_shape, 
        #             step=self.tiling_step, 
        #             save_root= save_folder_labels, 
        #             level = self.tiling_level,
        #             show = self.tiling_show,
        #             verbose = self.verbose)        
        # tiler(target_format=target_format)
        # tiler.test_show_image_labels()
        # self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ✅    ########################")


        return
    
    def _rename_tiff2tif(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.tiff' in file]
        old_new_names = [(file, file.replace('.tiff', '.tif')) for file in files ]
        # self.log.info(f"files:{old_new_names}")
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return
    

    

    def __call__(self) -> None:

        # 1) create tiles branch
        self._make_tiles_branch()
        # 1) split data
        self._split_data()
        # 2) prepare for tiling 
        self._move_slides_forth()
        # 3) tile images and labels:
        self.tile_dataset()
        #4) move slides back 
        self._move_slides_back()
        # 4) clean dataset, e.g. 
        self._clean_muw_dataset()

        return


def test_ProcessorManager(): 


    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    # DEVELOPMENT 
    src_root = '/Users/marco/Downloads/test_folders/test_hubmap_processor' if system == 'mac' else  r'D:\marco\datasets\slides'
    dst_root = '/Users/marco/Downloads/test_folders/test_hubmap_processor' if system == 'mac' else  r'D:\marco\datasets\slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.7, 0.15, 0.15]    
    data_source = 'hubmap'
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (1024,1024)
    tiling_step = 512
    tiling_level = 3
    tiling_show = True

    manager = ManagerBase(data_source=data_source,
                               src_root=src_root,
                               dst_root=dst_root,
                               slide_format=slide_format,
                               label_format=label_format,
                               split_ratio=split_ratio,
                               tiling_shape=tiling_shape,
                               tiling_step=tiling_step,
                               task=task,
                               tiling_level=tiling_level,
                               tiling_show=tiling_show,
                               verbose=verbose,
                               safe_copy=safe_copy)
    manager()

    return


if __name__ == '__main__': 
    test_ProcessorManager()