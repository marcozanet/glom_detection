import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from typing import List, Tuple, Literal
import numpy as np
import random
from conversion import Converter
from tiling import Tiler
from loggers import get_logger
from processor_tile import TileProcessor




class WSI_Processor(TileProcessor):

    def __init__(self,
                src_root: str, 
                dst_root: str, 
                step: int,
                convert_from: Literal['json_wsi_mask', 'jsonliketxt_wsi_mask', 'gson_wsi_mask'], 
                convert_to: Literal['json_wsi_bboxes', 'txt_wsi_bboxes'],
                slide_format: Literal['tiff', 'tif'],
                # label_format: Literal['json', 'gson'],
                ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                safe_copy: bool = True,
                tile_shape: tuple = (4096, 4096),
                verbose: bool = False,
                empty_perc: float = 0.1, 
                ) -> None:

        """ Sets paths and folders starting from tiles. 
            NB images should be named <name>.png <name-labelled>.png img_folder:"""

        self.log = get_logger()

        assert isinstance(src_root, str), TypeError(f"src_root type is {type(src_root)}, but should be str.")
        assert os.path.isdir(src_root), ValueError(f"{src_root} is not a dir. ")
        assert isinstance(dst_root, str), TypeError(f"dst_root type is {type(dst_root)}, but should be str.")
        assert os.path.isdir(dst_root), ValueError(f"{dst_root} is not a dir. ")
        assert isinstance(ratio, List), TypeError(f"'ratio' should be left empty or be a list. ")
        try:    
            ratio = [float(value) if isinstance(value, str) else value for value in ratio]
        except:
            TypeError(f"Values in 'ratio' can't be converted to float.")
        assert len(ratio) == 3 and round(np.sum(np.array(ratio)), 2) == 1.0, ValueError(f"'ratio' should be a list of floats with sum 1, but has sum {np.sum(np.array(ratio))}." )
        assert task in ['segmentation', 'detection', 'both'], ValueError(f"'task'= {task} should be either segmentation, detection or both. ")
        assert isinstance(tile_shape, tuple), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."
        assert isinstance(step, int), f"'step' should be int."
        assert isinstance(safe_copy, bool), f"safe_copy should be boolean."
        assert isinstance(empty_perc, float), f"'empty_perc' should be a float between 0 and 1."
        assert 0<=empty_perc<=1, f"'empty_perc' should be a float between 0 and 1."
        assert slide_format in ['tiff', 'tif'], f"'slide_format'={slide_format} should be either 'tiff' or 'tif' format."
        # assert label_format in ['json', 'gson'], f"'label_format'={label_format} should be either 'json' or 'gson' format."

        self.src_root = src_root
        self.dst_root = dst_root
        self.ratio = ratio
        self.task = task 
        self.tile_shape = tile_shape
        self.verbose = verbose
        self.step = step
        self.safe_copy = safe_copy
        self.empty_perc = empty_perc
        self.slide_format = slide_format
        self.convert_from = convert_from
        self.convert_to = convert_to

    
    def get_yolo_labels(self) -> None:
        """ Converts .json WSI annotation file to .txt tile annotations suitable to be trained with YOLO.  """
        
        self.log.info(f"Getting YOLO label tiles: ⏳")

        # 1) Conversion
        folder = self.src_root
        converter = Converter(folder = folder, 
                              convert_from=self.convert_from, 
                              convert_to=self.convert_to )
        converter()

        # 2) Label Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= self.tile_shape, 
                      save_root=self.src_root, 
                      step = self.step,
                      verbose = self.verbose)
        tiler(target_format='txt')

        self.log.info(f"Tiled annotations into .txt tiled files: ✅")
        labels_dir = os.path.join(self.src_root, 'labels')

        return labels_dir
    

    def get_yolo_images(self) -> None:
        """ Tiles the WSI and saves the patches in 'save_folder'. """

        self.log.info(f"Getting YOLO image tiles: ⏳")

        # Image Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= self.tile_shape, 
                      save_root=self.src_root, 
                      step = self.step,
                      verbose = self.verbose)
        tiler(target_format=self.slide_format)

        self.log.info(f"Tiled slide into image tiles: ✅")
        images_dir = os.path.join(self.src_root, 'images')
        
        return images_dir
    

    def _get_yolo_data(self) -> None:
        """ Tiles both WSI and its annotations and saves them into 'images', 'labels' folder.  """
        
        print(f"Tip: Make sure 'step' divides in tiles such that gloms are at least once fully captured in one tile. ")

        images_dir = self.get_yolo_images()
        labels_dir = self.get_yolo_labels()

        return images_dir, labels_dir
    


    def __call__(self) :

        # 1) clear existing datasets/make new one:
        self._clear_dataset()

        # 2) process images and labels:
        if self.task == 'detection':
            _, _ = self._get_yolo_data() 
        
        elif self.task == 'segmentation':
            print(f"❌ Splitting for segmentation task has not yet been implemented.")
            raise NotImplementedError()

        return 


def test_WSI_Processor():
    
    print(" ########################    TEST 1: ⏳     ########################")
    # setting:
    src_root = '/Users/marco/Downloads/test_folders/test_processor/send_windows'
    dst_root = '/Users/marco/Downloads/test_folders/test_processor/send_windows'
    ratio = [0.7, 0.15, 0.15]
    task = 'detection'
    convert_from = 'gson_wsi_mask'
    convert_to = 'txt_wsi_bboxes'    
    empty_perc =  0.1 
    step = 1024
    slide_format= 'tif' 
    tile_shape = (4096, 4096)
    verbose = False
    # make dirs:
    os.makedirs(src_root, exist_ok=True)
    os.makedirs(dst_root, exist_ok=True)
    # run testing
    processor = WSI_Processor(src_root = src_root, 
                              dst_root = dst_root,
                              convert_from = convert_from, 
                              convert_to = convert_to,                               
                              ratio = ratio, 
                              slide_format= slide_format,
                              task = task, 
                              step = step,
                              tile_shape = tile_shape,
                              empty_perc = empty_perc,
                              verbose = verbose)
    processor()
    print(" ########################    TEST 2: ✅    ########################")

    return  



if __name__ == '__main__':
    test_WSI_Processor()
