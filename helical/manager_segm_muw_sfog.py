from typing import Literal, Tuple
import os
from loggers import get_logger
from decorators import log_start_finish
from splitter import Splitter
from move_data import move_slides_for_tiling, move_slides_back_from_tiling
from tiler_segm_hub import TilerSegm
from cleaner_hubmap_segm import CleanerSegmHubmap
from manager_base import ManagerBase
from cleaner_muw_segm import CleanerSegmMuw
from converter_muw import ConverterMuW



class ManagerSegmMuw(ManagerBase): 
    def __init__(self,
                 map_classes:dict, 
                 inflate_points_ntimes: int,
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.map_classes = map_classes
        self.inflate_points_ntimes = inflate_points_ntimes

        #self.log.info(f"{self.__class__.__name__}.{'tile_dataset'}: Tiling dataset at '{self.wsi_dir}'")

        return
    
    def _rename_mrxsgson2gson(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.mrxs.gson' in file]
        old_new_names = [(file, file.replace('.mrxs.gson', '.gson')) for file in files ]
        # self.log.info(f"files:{old_new_names}")
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return
    
    def _parse(self): 

        self.data_source == 'muw', self.log.error(f"{self.__class__.__name__}.{'_parse'}: data_source is {self.data_source} but should be 'muw''")

        return

    
    def _clean_muw_dataset(self, safe_copy:bool=False) -> None: 
        """ Uses the dataset cleaner to finalize the dataset, e.g. by grouping classes 
            from {0:glom_healthy, 1:glom_na, 2: glom_sclerosed, 3: tissue}
            to {0:glom_healthy, 1:glom_sclerosed} """
        
        cleaner = CleanerSegmMuw(data_root=os.path.join(self.dst_root, self.task), 
                                    safe_copy=safe_copy,
                                    wsi_images_like = '*.tif', 
                                    wsi_labels_like = '*.json',
                                    tile_images_like = '*.png',
                                    tile_labels_like = '*.txt')
        cleaner()
        
        return
    

    def __call__(self) -> None:

        self._rename_mrxsgson2gson()
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
    

    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        """ Tiles a single folder"""
        class_name = self.__class__.__name__
        func_name = '_tile_folder'

        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        save_folder_labels = os.path.join(self.tiles_dir, dataset)
        save_folder_images = os.path.join(self.tiles_dir, dataset)


        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        if self.data_source == 'hubmap':
            self.log.error('dataset is hubmap but should be muw')

        elif self.data_source == 'muw': 

            converter = ConverterMuW(folder = slides_labels_folder, 
                                     stain = self.stain,
                                    convert_from='gson_wsi_mask',  
                                    convert_to='txt_wsi_bboxes',
                                    save_folder= slides_labels_folder, 
                                    level = self.tiling_level,
                                    verbose=self.verbose)
            converter()

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
    


