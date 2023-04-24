from typing import Literal, Tuple
import os
from loggers import get_logger
from decorators import log_start_finish
from splitter import Splitter
from move_data import move_slides_for_tiling, move_slides_back_from_tiling
from tiler_segm_hub import TilerSegm
from cleaner_hubmap_segm import CleanerSegmHubmap
from manager_base import ManagerBase



class ManagerSegmHubPAS(ManagerBase): 
    
    def __init__(self,
                 map_classes:dict, 
                 inflate_points_ntimes: int,
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.map_classes = map_classes
        self.inflate_points_ntimes = inflate_points_ntimes

        # self.log = get_logger()

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
        try:
            self.tile_dataset()
        except:
            self.log.error("Some error with tile_dataset (often times is the empty test folder during testings)")
        # 4) move slides back 
        self._move_slides_back()
        # 4) clean dataset, e.g. 
        self._clean_hubmap_dataset()

        return


    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        """ Tiles a single folder"""

        class_name = self.__class__.__name__
        func_name = '_tile_folder'

        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        # save_folder_labels = os.path.join(self.tiles_dir, dataset)
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

        return
    


