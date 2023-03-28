from typing import Literal, Tuple
import os
from loggers import get_logger
from decorators import log_start_finish
from splitter import Splitter
from move_data import move_slides_for_tiling, move_slides_back_from_tiling
from tiler_segm_hub import TilerSegm
from cleaner_hubmap_segm import CleanerSegmHubmap



class ManagerSegmHubPAS(): 
    def __init__(self,
                data_source: Literal['muw', 'hubmap'],
                src_root: str, 
                dst_root: str, 
                map_classes: dict,
                slide_format: Literal['tif', 'tiff'],
                label_format: Literal['gson', 'mrxs.gson'],
                tiling_shape: Tuple[int],
                tiling_step: int,
                tiling_level: int,
                inflate_points_ntimes:int = None,
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
        self.tiling_level = tiling_level
        self.tiling_show = tiling_show
        self.split_ratio = split_ratio
        self.task = task
        self.verbose = verbose
        self.safe_copy = safe_copy
        self.reproducibility = reproducibility
        self.wsi_dir = os.path.join(dst_root, self.task, 'wsi')
        self.tiles_dir = os.path.join(dst_root, self.task, 'tiles')
        self.data_source = data_source
        self.map_classes = map_classes
        self.inflate_points_ntimes = inflate_points_ntimes

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
        # splitter.move_already_tiled(tile_root = '/Users/marco/Downloads/muw_slides')
        # splitter._remove_empty_images()
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
    
    
    def _clean_hubmap_dataset(self, safe_copy:bool=False) -> None: 
        """ Uses the dataset cleaner to finalize the dataset, e.g. by grouping classes 
            from {0:glom_healthy, 1:glom_na, 2: glom_sclerosed, 3: tissue}
            to {0:glom_healthy, 1:glom_sclerosed} """
        
        cleaner = CleanerSegmHubmap(data_root=os.path.join(self.dst_root, self.task), 
                                    safe_copy=safe_copy,
                                    wsi_images_like = '*.tif', 
                                    wsi_labels_like = '*.json',
                                    tile_images_like = '*.png',
                                    tile_labels_like = '*.txt')
        cleaner()
        

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
        self._clean_hubmap_dataset()


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

        # if os.path.isdir(new_datafolder):
        #     shutil.rmtree(path = new_datafolder)
        #     print(f"Dataset at: {new_datafolder} removed.")
        
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


        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        if self.data_source == 'muw':
            self.log.error(NotImplementedError())

        elif self.data_source == 'hubmap': 
            tiler = TilerSegm(folder = slides_labels_folder,
                              map_classes=self.map_classes,
                              tile_shape= self.tiling_shape, 
                              step=self.tiling_step, 
                              save_root= save_folder_images, 
                              inflate_points_ntimes=self.inflate_points_ntimes,
                              level = self.tiling_level,
                              show = self.tiling_show,
                              verbose = self.verbose) 
        target_format = 'tif'
        tiler(target_format=target_format)
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

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
    


