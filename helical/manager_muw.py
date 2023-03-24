
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
# from cleaner import Cleaner
from manager_base import ManagerBase



class ManagerMUW(ManagerBase): 

    def __init__(self,
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        assert self.data_source == 'muw', self.log.error(ValueError(f"'data_source' is {self.data_source} but Manager used is 'ManagerMUW'"))
        self.data_source == "muw"

        return

    
    def _rename_mrxsgson2gson(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.mrxs.gson' in file]
        old_new_names = [(file, file.replace('.mrxs.gson', '.gson')) for file in files ]
        # self.log.info(f"files:{old_new_names}")
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

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
        if self.data_source == 'muw':
            converter = ConverterMuW(folder = slides_labels_folder, 
                                     stain = self.stain,
                                    convert_from='gson_wsi_mask',  
                                    convert_to='txt_wsi_bboxes',
                                    save_folder= slides_labels_folder, 
                                    level = self.tiling_level,
                                    verbose=self.verbose)
        elif self.data_source == 'hubmap':
            converter = ConverterHubmap(folder = slides_labels_folder, 
                                        stain = self.stain,
                                        convert_from='json_wsi_mask',  
                                        convert_to='txt_wsi_bboxes',
                                        save_folder= slides_labels_folder, 
                                        level = self.tiling_level,
                                        verbose=self.verbose)
        converter()
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ✅    ########################")


        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        if self.data_source == 'muw':
            tiler = Tiler(folder = slides_labels_folder, 
                        tile_shape= self.tiling_shape, 
                        step=self.tiling_step, 
                        save_root= save_folder_images, 
                        level = self.tiling_level,
                        show = self.tiling_show,
                        verbose = self.verbose)
        elif self.data_source == 'hubmap': 
            print(f"alling tiler hubmap")
            tiler = TilerHubmap(folder = slides_labels_folder, 
                                tile_shape= self.tiling_shape, 
                                step=self.tiling_step, 
                                save_root= save_folder_images, 
                                level = self.tiling_level,
                                show = self.tiling_show,
                                verbose = self.verbose)
        target_format = 'tif'
        tiler(target_format=target_format)
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

        # 3) tile labels:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ⏳    ########################")
        target_format = 'txt'
        tiler = Tiler(folder = slides_labels_folder, 
                    tile_shape= self.tiling_shape, 
                    step=self.tiling_step, 
                    save_root= save_folder_labels, 
                    level = self.tiling_level,
                    show = self.tiling_show,
                    verbose = self.verbose)        
        tiler(target_format=target_format)
        tiler.test_show_image_labels()
        self.log.info(f"{class_name}.{func_name}: ######################## TILING LABELS: ✅    ########################")


        return

    def __call__(self) -> None:


        self._rename_tiff2tif()
        self._rename_mrxsgson2gson()
        # 1) create tiles branch
        self._make_tiles_branch()
        # 1) split data
        self._split_data()
        # 2) prepare for tiling 
        self._move_slides_forth()
        # 3) tile images and labels:
        self.tile_dataset()
        # 4) move slides back 
        self._move_slides_back()
        # 5) clean dataset, e.g. 
        # self._clean_muw_dataset()


        return
    



def test_ManagerMUW(): 


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

    manager = ManagerMUW(data_source=data_source,
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
    test_ManagerMUW()