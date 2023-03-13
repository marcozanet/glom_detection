
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
from cleaner import Cleaner



class ManagerHubmap(): 
    def __init__(self,
                src_root: str, 
                dst_root: str, 
                slide_format: Literal['tif', 'tiff'],
                label_format: Literal['gson', 'mrxs.gson'],
                tiling_shape: Tuple[int],
                tiling_step: int,
                tiling_show: bool = True,
                split_ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                safe_copy: bool = False,
                verbose: bool = False,
                empty_perc: float = 0.1,
                reproducibility: bool = True) -> None:
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.slide_format = slide_format
        self.label_format = label_format
        self.tiling_shape = tiling_shape
        self.tiling_step = tiling_step
        self.tiling_show = tiling_show
        self.split_ratio = split_ratio
        self.task = task
        self.verbose = verbose
        self.safe_copy = safe_copy
        self.reproducibility = reproducibility
        self.wsi_dir = os.path.join(dst_root, self.task, 'wsi')
        self.tiles_dir = os.path.join(dst_root, self.task, 'tiles')
        

        self.log = get_logger()

        return

    def _split_data(self):

        print(" ########################    SPLITTING DATA: ⏳     ########################")

        @log_start_finish(class_name=self.__class__.__name__, func_name='_split_data', msg= f" Splitting slides")
        def do():
            splitter = Splitter(src_dir=self.src_root,
                                dst_dir=self.dst_root,
                                image_format=self.slide_format,
                                ratio=self.split_ratio,
                                task=self.task,
                                verbose = self.verbose, 
                                safe_copy = self.safe_copy,
                                reproducibility=self.reproducibility)
            splitter()
            return
        
        do()
        print(" ########################    SPLITTING DATA 2: ✅    ########################")

        return
    
    def _move_slides_forth(self): 
        """ Moves slides together with labels to be ready for tiling. """

        move_slides_for_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        self.log.info(f"{self.__class__.__name__}.{'_move_slides_forth'}: Moved slides forth.")

        return
    
    def _move_slides_back(self):

        move_slides_back_from_tiling(wsi_folder=self.wsi_dir, slide_format=self.slide_format)
        self.log.info(f"{self.__class__.__name__}.{'_move_slides_back'}: Moved slides forth.")

        return
    
    
    def _clean_hubmap(self, safe_copy:bool=False) -> None: 
        """ Uses the dataset cleaner to finalize the dataset, e.g. by grouping classes 
            from {0:glom_healthy, 1:glom_na, 2: glom_sclerosed, 3: tissue}
            to {0:glom_healthy, 1:glom_sclerosed} """
        
        cleaner = Cleaner(data_root=os.path.join(self.dst_root, self.task),
                         safe_copy=safe_copy,
                         wsi_images_like = '*.tif', 
                         wsi_labels_like = '*.txt',
                         tile_images_like = '*_*_*.png',
                         tile_labels_like = '*_*_*.txt')
        cleaner._clean_hubmap()
        

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
        # self._move_slides_back()
        # 4) clean dataset, e.g. 
        # self._clean_hubmap()


        return
    

    def tile_dataset(self): 
        """ Tiles the whole dataset: train, val, test. """

        self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: Tiling dataset at '{self.wsi_dir}'")
        datasets = ['train', 'val', 'test']
        for dataset in datasets: 
            self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: Tiling '{dataset}'")
            self._tile_folder(dataset=dataset)

        return
    
    
    def _make_tiles_branch(self): 
        """ Creates a tree for tiles with same structures as the one for wsi: tiles -> train,val,test -> images,labels"""

        # 1) makedirs:
        self.log.info(f"{self.__class__.__name__}.{'_make_tiles_branch'}: Creating new tiles branch at '{self.tiles_dir}'")
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(self.tiles_dir, subfold, subsubfold), exist_ok=True)

        return         
    

    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        """ Tiles a single folder"""
        class_name = self.__class__.__name__
        func_name = '_tile_folder'

        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        save_folder_labels = os.path.join(self.tiles_dir, dataset)
        save_folder_images = os.path.join(self.tiles_dir, dataset)

        # 1) convert annotations to yolo format:
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ⏳    ########################")
        converter = ConverterHubmap(folder = slides_labels_folder, 
                                    map_classes = {'glomerulus':0},
                                    convert_from='json_wsi_mask',  
                                    convert_to='txt_wsi_bboxes',
                                    save_folder= slides_labels_folder, 
                                    level = 0,
                                    verbose=self.verbose)
        converter()
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ✅    ########################")


        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        tiler = TilerHubmap(folder = slides_labels_folder, 
                            tile_shape= self.tiling_shape, 
                            step=self.tiling_step, 
                            save_root= save_folder_images, 
                            # clean_every_file = True,
                            # cleaner_path = os.path.join(self.dst_root, self.task),
                            level = 0,
                            show = self.tiling_show,
                            verbose = self.verbose)
        target_format = 'tif'
        tiler(target_format=target_format)
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

        # 3) tile labels:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ⏳    ########################")
        target_format = 'txt'
        tiler = TilerHubmap(folder = slides_labels_folder, 
                            tile_shape= self.tiling_shape, 
                            step=self.tiling_step, 
                            save_root= save_folder_labels, 
                            level = 0,
                            show = self.tiling_show,
                            verbose = self.verbose)        
        tiler(target_format=target_format)
        # tiler.test_show_image_labels()
        self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ✅    ########################")


        return
    



def test_ProcessorManager(): 


    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    # DEVELOPMENT 
    src_root = '/Users/marco/helical_tests/test_hubmap_manager' if system == 'mac' else  r'D:\marco\datasets\slides'
    dst_root = '/Users/marco/helical_tests/test_hubmap_manager' if system == 'mac' else  r'D:\marco\datasets\slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.7, 0.15, 0.15]    
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (2048,2048)
    tiling_step = 512
    tiling_show = False

    manager = ManagerHubmap(src_root=src_root,
                            dst_root=dst_root,
                            slide_format=slide_format,
                            label_format=label_format,
                            split_ratio=split_ratio,
                            tiling_shape=tiling_shape,
                            tiling_step=tiling_step,
                            task=task,
                            tiling_show=tiling_show,
                            verbose=verbose,
                            safe_copy=safe_copy)
    manager()

    return


if __name__ == '__main__': 
    test_ProcessorManager()