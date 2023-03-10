
import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from glob import glob
from typing import List, Tuple
import warnings
from patchify import patchify
import numpy as np 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io, draw
import cv2
import geojson
from typing import Literal
import random
from loggers import get_logger
from decorators import log_start_finish
import json
import cv2
from tiling import Tiler



class TilerHubmap(Tiler): 

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        return
    
    def _get_tile_images(self, 
                        fp: str, 
                        overlapping: bool = False,
                        save_folder: str = None) -> None:
        """ Tiles the WSI into tiles and saves them into the save_folder. """
        
        class_name = self.__class__.__name__
        func_name = '_get_tile_images'

        assert os.path.isfile(fp), ValueError(f"{fp} is not a valid filepath.")
        assert isinstance(self.tile_shape, tuple) and len(self.tile_shape) == 2, TypeError(f"'tile_shape':{self.tile_shape} should be a tuple of two int.")
        assert isinstance(self.tile_shape[0], int) and isinstance(self.tile_shape[1], int), TypeError(f"'tile_shape':{self.tile_shape} should be a tuple of two int.")
        assert isinstance(overlapping, bool), TypeError(f"'overlapping' should be a boolean. ")
        save_folder = os.path.join(self.save_root, 'images') if save_folder is None else save_folder
       
        @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Tiling image: '{os.path.basename(fp)}'" )
        def do():        
                
            if self._is_slide_computed(fp):
                return

            w, h = self.tile_shape

            # 1) read slide:
            try:
                self.log.info(f"{class_name}.{func_name}: ⏳ Opening '{os.path.basename(fp)}':" )
                slide = openslide.OpenSlide(fp)
            except:
                self.log.error(f"{class_name}.{func_name}: ❌ Couldn t open file: '{os.path.basename(fp)}'. Skipping." )
                return
            self.log.info(f"{class_name}.{func_name}: ✅ Opening '{os.path.basename(fp)}':" )
            W, H = slide.dimensions

            # 1) reading region:
            self.log.info(f"{class_name}.{func_name}: ⏳ Reading slide with shape ({W, H}):")
            assert self.level == 0 , self.log.error(f"self.level:{self.level}, but hubmap slides are not pyramidal.")
            try:
                region = slide.read_region(location = (0,0) , level = self.level, size= (W,H)).convert("RGB")
            except:
                self.log.error(f"{class_name}.{func_name}: ❌ Reading region failed")

            # 2) converting to numpy array:
            self.log.info(f"{class_name}.{func_name}: ⏳ Converting to numpy:")
            try:
                np_slide = np.array(region)
            except:
                self.log.error(f"{class_name}.{func_name}: ❌ Conversion to numpy.")
            self.log.info(f"{class_name}.{func_name}: ✅ Conversion to numpy.")

            # 3) patchification:
            self.log.info(f"{class_name}.{func_name}: ⏳ Patchifying:")
            try:
                if overlapping is False:
                    patches = patchify(np_slide, (w, h, 3), step =  self.step )
                    w_tiles,h_tiles = patches.shape[0],patches.shape[1]
                    sample_fn = os.path.split(fp.replace('.tif', ''))[1]
                    self._write_ntiles(sample_fn=sample_fn, dims=(w_tiles,h_tiles))
                else:
                    raise NotImplementedError()
            except:
                self.log.error(f"{class_name}.{func_name}: ❌ Patchifying.")
            self.log.info(f"{class_name}.{func_name}: ✅ Patchifying.")

            # 3) save patches:
            self.log.info(f"{class_name}.{func_name}: ⏳ Saving patches:")
            patches = patches[:, :, 0, ...]
            for i in tqdm(range(patches.shape[0]), desc= f"⏳ Tiling"):
                for j in range(patches.shape[1]):
                    save_fp = fp.replace(f'.{self.format}',f'_{i}_{j}.png') if self.multiple_samples else fp.replace(f'.{self.format}',f'_{i}_{j}.png')
                    if save_folder is not None:
                        fname = os.path.split(save_fp)[1]
                        save_fp = os.path.join(save_folder, fname)

                    cv2_img = cv2.cvtColor(patches[i, j], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_fp, img = cv2_img)
            self.log.info(f"{class_name}.{func_name}: ✅ Saved patches in {save_folder}.")

            return
        
        do()

        return

    def _get_tile_labels(self, fp: str, save_folder: str = None ):
        ''' Makes tile txt annotations in YOLO format (normalized) out of (not normalized) txt annotations for the entire image.
            Annotations tiles are of shape 'tile_shape' and are only made around each object contained in the WSI annotation, since YOLO doesn't 
            need annotations for empty images. 
            fp = path to WSI (not normalized) annotation in .txt format '''
        
        assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")
        
        class_name = self.__class__.__name__
        func_name = '_get_tile_labels'
        save_folder = os.path.join(self.save_root, 'labels') if save_folder is None else save_folder

        @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Tiling label: '{os.path.basename(fp)}'" )
        def do():
            
            self.log.info(f"{class_name}.{'_get_tile_labels'}: Tiliing label '{fp}'") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.txt
            wsi_fn = os.path.split(fp)[1].split('.')[0]
            ret = self.n_tiles[wsi_fn]
            x_max, y_max  = self.n_tiles[wsi_fn][0], self.n_tiles[wsi_fn][1]
            self.log.info(f"{self.__class__.__name__}.{'_get_tile_labels'}: x_max:{x_max}, y_max:{y_max}")

            # Get BB from txt file:
            assert os.path.isfile(fp), f"{class_name}.{func_name}:'fp' is not a valid filepath"
            with open(fp, 'r') as f:
                text = f.readlines()
                f.close()

            for row in text:

                # get values:
                items = row.split(sep = ',')
                xc, yc, box_w, box_h = [float(num) for num in items[1:]]

                clss = items[0]
                W, H = self.tile_shape[0], self.tile_shape[1]
                
                x_start = xc - box_w / 2 # e.g. 0 - 
                x_end = xc + box_w / 2
                y_start = yc - box_h / 2
                y_end = yc + box_h / 2

                for i in range(0, x_max*W, self.step):
                    if i <=  x_start <=  i + W or i <=  x_end <=  i + W:
                        for j in range(0, y_max*H, self.step):
                            if j <=  y_start <=  j + H or j <=  y_end <=  j + H:
                                x0 = i if x_start <= i else x_start
                                x1 = i + W if x_end >= i + W else x_end
                                y0 = j if y_start <= j else y_start
                                y1 = j + H if y_end >= j + H else y_end 

                                tile_xc = (x0 + x1)/2 - i  # no need to normalize, self._write_txt does that
                                tile_yc = (y0 + y1)/2 - j 
                                tile_w = (x1 - x0) 
                                tile_h = (y1 - y0) 

                                assert 0 <= tile_xc <=W, f"{x0, x1, i, tile_xc}"
                                assert 0 <= tile_xc <=W, f"'tile_xc'={tile_xc}, but should be in  (0,{W})."
                                assert 0 <= tile_yc <=H, f"'tile_yc'={tile_yc}, but should be in  (0,{H})."
                                assert 0 <= tile_w <=W, f"'tile_w'={tile_w}, but should be in  (0,{W})."
                                assert 0 <= tile_h <=H, f"'tile_h'={tile_h}, but should be in  (0,{H})."

                                # save
                                save_fp = fp.replace('.txt', f'_{j//self.step}_{i//self.step}.txt') # img that contains a part of the glom
                                if save_folder is not None:
                                    fname = os.path.split(save_fp)[1]
                                    save_fp = os.path.join(save_folder, fname)
                                self._write_txt(clss, tile_xc, tile_yc, tile_w, tile_h, save_fp)

            self.log.info(f"{class_name}.{func_name}: ✅ Tile labels saved in '{save_folder}'." )

            return
        
        do()

        return
    

    def __call__(self, target_format: str, save_folder: str = None) -> None:
        """ Tiles/patchifies WSI or annotations. """

        SLIDE_FORMATS =  ['tiff', 'tif']
        LABEL_FORMATS = ['txt']
        class_name = self.__class__.__name__
        func_name =  '__call__'

        assert target_format in SLIDE_FORMATS or target_format in LABEL_FORMATS, ValueError(f"Patchification target format = {target_format} should be either an image in 'tiff', 'tif' format or an annotation in 'txt' format. ")
        assert save_folder is None or os.path.isdir(save_folder), ValueError(f"'save_folder':{save_folder} should be either None or a valid dirpath. ")
        
        self.json_ntile = os.path.join(self.folder, 'n_tiles.json')
        default_folder = 'images' if (target_format == 'tiff' or target_format == 'tif') else 'labels'
        save_folder = os.path.join(self.save_root, default_folder) if save_folder is None else save_folder
        self.format = target_format
        target = "IMAGES" if (self.format == 'tif' or self.format == 'tiff') else "LABELS"

        self.log.info( f"{class_name}.{func_name}: ⏳ START TILING {target} from folder:'{self.folder}'. Results will be saved in '{save_folder}'.")
        
        # 1) make save folders:
        os.makedirs(save_folder, exist_ok=True)

        # 2) get WSI/annotations:
        files = self._get_files(format = target_format)
        fnames = [os.path.split(file)[1].split('.')[0] for file in files]

        # 3) tile files:
        if len(files) == 0: 
            self.log.error(f"{class_name}.{func_name}: ❌ No file in format '{target_format}' was found in '{self.folder}'.")

        # check if all tiling is complete:
        _, uncompleted = self._get_completed_files(fnames)
        if len(uncompleted) == 0: 
            self.log.info(f"{class_name}.{func_name}: 🎉 All files in folder '{os.path.dirname(files[0])}' have been tiled and saved in '{save_folder}'.  ")
            if self.show is True:
                self.test_show_image_labels()
            return
        
        if target_format == 'txt':
            self.log.info(f"calling get_n_tiles")
            self.n_tiles = self._get_n_tiles() #(files, overlapping=False, save_folder=save_folder)
            self.log.info(f"self.n_tiles:{self.n_tiles}")
        

        for file in files:
            
            # clean folder from small files: 
            self._remove_small_files(files=files)
            # tile label:
            if target_format == 'txt':
                # check if this tile is already tiled:
                if self._is_sample_computed(sample_fp=file):
                    self.log.info(f"{class_name}.{func_name}: '{os.path.split(file.split('.')[0])[1]}' already tiled. Skipping. ")
                    continue
                self._get_tile_labels(fp = file, save_folder=save_folder)
            # tile image:
            if target_format == 'tif':
                # check if computed is inside get_tile_images func
                self._get_tile_images(fp = file, save_folder=save_folder )



        return

    def _is_slide_computed(self, sample_fp:str):
        """ Returns whether the sample file was already computed. """
        class_name = self.__class__.__name__
        func_name = '_is_slide_computed'

        format = 'png' if (self.format == 'tiff' or self.format == 'tif') else 'txt'
        save_folder = os.path.join(self.save_root, 'images') if format == 'png' else os.path.join(self.save_root, 'labels')
        basename = os.path.split(sample_fp.split('.')[0])[1] # e.g. 200104066_09_SFOG_sample0
        name_like = os.path.join(save_folder, basename) # e.g. /Users/marco/Downloads/test_folders/test_tiler/test_1slide/images/200104066_09_SFOG_sample0
        name_like = f'{name_like}*.{format}' # e.g. /Users/marco/Downloads/test_folders/test_tiler/test_1slide/images/200104066_09_SFOG_sample0*.png
        matching_files = glob(name_like)
        computed = False if len(matching_files) <= 8 else True

        if computed:
            self.log.warning(f"{class_name}.{func_name}: 😪 Tiler: found .{format} tiles in '{save_folder}' for {os.path.split(basename)[0]}.{format}. Skipping sample.")

        return computed