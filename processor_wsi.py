import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import yaml
from typing import List, Tuple, Literal
import numpy as np
import shutil
from tqdm import tqdm
from skimage import measure, io, color, draw, transform
import time
import matplotlib.pyplot as plt
import random
import geojson


class WSI_Processor():

    def __init__(self,
                src_root: str, 
                dst_root: str, 
                ratio = [0.7, 0.15, 0.15], 
                mode = Literal['detection', 'segmentation', 'both'],
                empty_perc: float = 0.1) -> None:

        """ Sets paths and folders starting from tiles. 
            NB images should be named <name>.png <name-labelled>.png img_folder:"""
        
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
        assert mode in ['segmentation', 'detection', 'both'], ValueError(f"'mode'= {mode} should be either segmentation, detection or both. ")

        self.src_root = src_root
        self.dst_root = dst_root
        self.ratio = ratio
        self.mode = mode 
        self.empty_perc = 0.1 if empty_perc is None else empty_perc
    

    def _read_slide(self, fp) -> Tuple[int, int]:
        ''' Reads the slide and returns dims. '''

        assert os.path.isfile(fp), ValueError(f"WSI path: '{fp}' is not a valid filepath. ")
        try:
            wsi = openslide.OpenSlide(fp)
            W, H = wsi.dimensions
        except:
            print(f"Couldn't open {fp}")

        return W, H

    def get_bounding_boxes(self, W, H, fp, returned = 'yolo'):
        ''' Iterates through gloms and gets the bounding box from segmentation geojson annotations. '''
        
        # read file
        with open(fp, 'r') as f:
            data = geojson.load(f)
            f.close()
            
        gloms = 0
        new_coords = []
        boxes = []

        # saving outer coords (bounding boxes) for each glom
        x_min = 10000000000
        y_min = 10000000000
        x_max = 0
        y_max = 0

        # access polygon vertices of each glom
        for glom in data:
            gloms += 1
            vertices = glom['geometry']['coordinates']
            
            # saving outer coords (bounding boxes) for each glom
            x_min = 10000000000
            y_min = 10000000000
            x_max = 0
            y_max = 0
            for i, xy in enumerate(vertices[0]):
                x = xy[0]
                y = xy[1]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max 
                y_min = y if y < y_min else y_min

            if x_max > W:
                raise Exception()
            if y_max > H:
                raise Exception()

            x_c =  (x_max + x_min) / 2 
            y_c = (y_max + y_min) / 2  
            box_w, box_y = (x_max - x_min) , (y_max - y_min)
            new_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]) 
            boxes.append([0, x_c, y_c, box_w, box_y])

        return_obj = boxes if returned == 'yolo' else new_coords

        return return_obj



    def _convert_json2txt() -> None:
        """ Converts .json annotations to .txt annotations in YOLO format with WSI coordinates. """

        raise NotImplementedError()

        return

    
    def _convert_json2txt(folder: str) -> None:
        """ Converts .json annotations to .txt segmentations in YOLO format. """

        raise NotImplementedError()

        return
    
    def _get_tiles_bboxes() -> None:
        """ Takes .txt segmentations of WSIs and converts them into .txt bounding box annotations """

        raise NotImplementedError()


        return
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    