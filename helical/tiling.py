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

class Tiler():

    def __init__(self, 
                folder: str, 
                step: int,
                tile_shape: tuple = (2048, 2048),
                save_root = None, 
                multiple_samples: bool = True,
                level:int = 0,
                show = False,
                # region_annotations: str = None,
                verbose: bool = False) -> None:
        """ Class for patchification/tiling of WSIs and annotations. """

        self.log = get_logger()
        assert os.path.isdir(folder), ValueError(f"Provided 'folder':{folder} is not a valid dirpath.")
        assert isinstance(tile_shape, tuple), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape' should be a tuple of int.")
        assert save_root is None or os.path.isdir(save_root), ValueError(f"'save_root':{save_root} should be either None or a valid dirpath. ")
        assert isinstance(step, int), f"'step' should be int."
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."
        assert isinstance(multiple_samples, bool), f"'multiple_samples' should be a boolean."
        assert isinstance(show, bool), f"'show' should be a boolean."
        # assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")


        self.folder = folder 
        self.tile_shape = tile_shape
        self.save_root = save_root
        self.step = step
        self.verbose = verbose
        self.multiple_samples = multiple_samples
        self.level = level
        self.show = show

        return
    


    def _get_tile_labels(self, fp: str, save_folder: str = None ):
        ''' Makes tile txt annotations in YOLO format (normalized) out of (not normalized) txt annotations for the entire image.
            Annotations tiles are of shape 'tile_shape' and are only made around each object contained in the WSI annotation, since YOLO doesn't 
            need annotations for empty images. 
            fp = path to WSI (not normalized) annotation in .txt format '''
        
        assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")
        
        # self.log.info(f"fp: {fp}")
        class_name = self.__class__.__name__
        func_name = '_get_tile_labels'
        save_folder = os.path.join(self.save_root, 'labels') if save_folder is None else save_folder

        @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Tiling label: '{os.path.basename(fp)}'" )
        def do():
            
            self.log.info(f"Tiliing label {fp}") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.txt
            # from tile folder I get the x possible values and y possible values:
            # dirname = os.path.split(tile_images_fp)[0].replace('labels', 'images') # i.e. label
            # superdirname = os.path.dirname(dirname)
            # save_folder = e.g. D:\marco\datasets\muw_retiled\wsi\test\labels 
            # tile_images_fp = os.path.join(os.path.dirname(save_folder), save_folder.replace('labels', 'images'))
            
            
            # tile_images_fp = save_folder.replace('labels', 'images')
            wsi_fn = os.path.split(fp)[1].split('.')[0]
            # files = [file for file in os.listdir(tile_images_fp) if '.png' in file and wsi_fn in file]

            self.log.info(f"trying to open self.ntiles[{wsi_fn}]")
            ret = self.n_tiles[wsi_fn]
            self.log.info(f"ret: {ret}, type: {type(ret)}")
            x_max, y_max  = self.n_tiles[wsi_fn][0], self.n_tiles[wsi_fn][1]
            self.log.info(f"x_max:{x_max}, y_max:{y_max}")
            # x_max, y_max = x_max -1 , y_max -1
            # num_x_tiles = [int(file.split('_')[-2]) for file in files]
            # num_y_tiles = [int(file.split('_')[-1].split('.')[0]) for file in files]
            # if len(num_x_tiles) == 0: 
            #     self.log.warning(f"{class_name}.{func_name}: ❌ No tile images found. Skipping tiling of annotations for '{wsi_fn}'", )
            #     return

            # x_max = max(num_x_tiles)
            # y_max = max(num_y_tiles)

            # Get BB from txt file:
            with open(fp, 'r') as f:
                text = f.readlines()
                f.close()
            
            # raise NotImplementedError()
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
    
    
    def _write_ntiles(self, sample_fn:str, dims:tuple):

        new_data = {sample_fn:list(dims)}
        to_json = {}
        if os.path.isfile(self.json_ntile):
            with open(self.json_ntile, 'r') as f:
                self.log.info('OPNEEEEEEEEEE')
                from_json = json.load(f)
            self.log.info(f"type: {from_json}")
            to_json.update(from_json)
        
        to_json.update(new_data)
        self.log.info(to_json)
        with open(self.json_ntile, 'w') as f: 
            json.dump(to_json, fp = f)

        
        return
    
    def _get_n_tiles(self): 

        assert os.path.isfile(self.json_ntile), f"'json_ntile':{self.json_ntile} file is not a valid filepath."
        self.log.info("assert passed")
        with open(self.json_ntile, 'r') as f:
            n_tiles = json.load(f)
        self.log.info("get_n_tiles opened")
        
        # files = [os.path.split(fp.split('.')[0])[1] for fp in files]
        # assert all([os.path.isfile(fp) for fp in files]), ValueError(f"some path in {files} is not a valid filepath.")
        # txt_files = glob(os.path.join(self.folder, '*_sample*.txt'))
        # keys = [os.path.split(file)[1].split('.')[0] for file in txt_files]

        # for key in keys:


        return n_tiles
    





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

            # 2) if file has multi_samples -> region = sample:
            if self.multiple_samples is True:
                # get file with location of image/label samples within the slide:
                multisample_loc_file = self._get_multisample_loc_file(fp, file_format='geojson')
                sample_locations = self._get_location_w_h(fp = multisample_loc_file) if multisample_loc_file is not None else [{'location':(0,0), 'w':W, 'h':H}]
            else:
                multisample_loc_file = None
                sample_locations = [{'location':(0,0), 'w':W, 'h':H}]


            for sample_n, sample in enumerate(sample_locations):
                
                location, W, H = sample['location'], sample['w'], sample['h']
                
                # 1) reading region:
                # self.log.info(f"fp:{fp}")
                self.log.info(f"{class_name}.{func_name}: ⏳ Reading region ({W, H}) of sample_{sample_n}:")
                try:
                    region = slide.read_region(location = location , level = self.level, size= (W,H)).convert("RGB")
                except:
                    self.log.error(f"{class_name}.{func_name}: ❌ Reading region failed")

                # 2) converting to numpy array:
                self.log.info(f"{class_name}.{func_name}: ⏳ Converting to numpy sample_{sample_n}:")
                try:
                    np_slide = np.array(region)
                except:
                    self.log.error(f"{class_name}.{func_name}: ❌ Conversion to numpy.")
                self.log.info(f"{class_name}.{func_name}: ✅ Conversion to numpy.")

                # 3) patchification:
                self.log.info(f"{class_name}.{func_name}: ⏳ Patchifying sample_{sample_n}:")
                try:
                    if overlapping is False:
                        patches = patchify(np_slide, (w, h, 3), step =  self.step )
                        w_tiles,h_tiles = patches.shape[0],patches.shape[1]
                        # self.log.info(f"dims:{(w_tiles,h_tiles)}")
                        sample_fn = os.path.split(fp.replace('.tif', f"_sample{sample_n}"))[1]
                        # self.log.info(f"sample_fn:{sample_fn}")
                        self._write_ntiles(sample_fn=sample_fn, dims=(w_tiles,h_tiles))
                    else:
                        raise NotImplementedError()
                except:
                    self.log.error(f"{class_name}.{func_name}: ❌ Patchifying.")
                self.log.info(f"{class_name}.{func_name}: ✅ Patchifying.")

                # 3) save patches:
                self.log.info(f"{class_name}.{func_name}: ⏳ Saving patches of sample_{sample_n}:")
                patches = patches[:, :, 0, ...]
                fname = f"{os.path.split(fp)[1]}, sample {sample_n+1}/{len(sample_locations)}" if multisample_loc_file is not None else os.path.split(fp)[1]
                for i in tqdm(range(patches.shape[0]), desc= f"⏳ Tiling '{fname}'"):
                    for j in range(patches.shape[1]):
                        save_fp = fp.replace(f'.{self.format}',f'_sample{sample_n}_{i}_{j}.png') if self.multiple_samples else fp.replace(f'.{self.format}',f'_{i}_{j}.png')
                        if save_folder is not None:
                            self.log.info(f"save_folder:{save_folder}")
                            fname = os.path.split(save_fp)[1]
                            self.log.info(f"fname:{fname}")
                            save_fp = os.path.join(save_folder, fname)
                            self.log.info(f"save_fp:{save_fp}")

                        cv2_img = cv2.cvtColor(patches[i, j], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_fp, img = cv2_img)
                        # pil_img = Image.fromarray(patches[i, j])
                        # pil_img.save(save_fp)
                self.log.info(f"{class_name}.{func_name}: ✅ Saved sample_{sample_n} patches in {save_folder}.")

            return
        
        do()

        return
    

    def _get_completed_files(self, files:list) -> Tuple[list, list]:
        """ Returns a list of computed and not-computed files. """

        completed = []
        uncompleted = []
        for file in files:
            if self._is_slide_computed(file):
                completed.append(file)
            else:
                uncompleted.append(file)
        
        self.log.info(f"{self.__class__.__name__}.{'_get_completed_files'}: Completed {len(completed)}/{len(completed)+len(uncompleted)} files.")
        
        return completed, uncompleted
    
    
    def _remove_small_files(self, files:list, mem_size_min: int = 500_000): # min size: 500kB
        
        image_folder = os.path.join(self.save_root, 'images')
        files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if "DS" not in file]
        # min = 100_000_000
        deleted = 0
        for file in tqdm(files, desc = "Removing small files"): 
            # min = os.path.getsize(file) if os.path.getsize(file) < min else min
            if os.path.getsize(file) < mem_size_min: 
                os.remove(path = file)
                deleted += 1
        
        self.log.info(f"{self.__class__.__name__}.{'_remove_small_files'}: Removed {deleted} small files (< {int(mem_size_min/1000)}kB)")


        return
    
    def _get_multi_txt(self, txt_file:str) -> list:
        """ Given a txt file like '200104066_09_SFOG.txt', it returns a list of matching txt samples/rois like: 
            [200104066_09_SFOG_sample0.txt, 200104066_09_SFOG_sample1.txt]"""
        
        basename = txt_file.split('.')[0]
        samples = glob(f"{basename}_sample*.txt")

        # raise NotImplementedError()
        

        return samples



    
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
                if self.multiple_samples is True: 
                    samples_txt = self._get_multi_txt(txt_file=file)
                    for sample in samples_txt:
                        # check if this tile is already tiled:
                        if self._is_sample_computed(sample_fp=sample):
                            continue
                        self._get_tile_labels(fp = sample, save_folder=save_folder)

            # tile image:
            if target_format == 'tif':
                # check if computed is inside get_tile_images func
                self._get_tile_images(fp = file, save_folder=save_folder )
                
            # # check if all tiling is complete:
            # _, uncompleted = self._get_completed_files(fnames)
            # if len(uncompleted) == 0: 
            #     self.log.info(f"{class_name}.{func_name}: 🎉 All files in folder '{os.path.dirname(file)}' have been tiled and saved in '{save_folder}'.  ")
            #     if self.show is True:
            #         self.test_show_image_labels()
            #     return


        return


    ############ HELPER FUNCTIONS ############

    def _write_txt(self, clss, xc, yc, box_w, box_h, save_fp):

        # write patch annotation txt file:
        w = self.tile_shape[0]
        h = self.tile_shape[1]
        text = f'{clss} {xc/w} {yc/h} {box_w/w} {box_h/h}\n'   # TODO DIVIDE TO NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # save txt file:
        with open(save_fp, 'a+') as f:
            f.write(text)
        
        return


    def _get_files(self, format:str ) -> List[str]:
        """ Collects source files to be converted. """
        class_name = self.__class__.__name__
        func_name = '_get_files'


        files = glob(os.path.join(self.folder, f'*.{format}' ))
        files = [file for file in files if "sample" not in file]

        # sanity check:
        already_patched = glob(os.path.join(self.folder, f'*_?_?.{format}' ))
        if len(already_patched) > 0:
            self.log.error(f"{class_name}.{func_name}: ❗️ Warning: found tile annotations (e.g. {already_patched[0]}) in source folder.")
        files = [file for file in files if file not in already_patched]

        return files
    

    def _get_multisample_loc_file(self, fp: str, file_format: Literal['geojson'], mode = 'images'):
        """ For multisample slides, it collects files with annotations of sample locations within the slides. """

        FORMATS = ['geojson']
        assert os.path.isfile(fp), f"'fp':{fp} if not a valid filepath."
        assert file_format in FORMATS, f"'file_format' should be one of {FORMATS}."
        assert self.multiple_samples is True, f"'multiple_samples' is False, but _get_multisample_annotations() was called. "

        # name = "msample_image" if self.format in ['tiff', 'tif'] else "sample_label"
        multisample_loc_file = fp.replace(f".{self.format}", f".{file_format}") if mode == 'images' else fp.replace(f".tif", f".{file_format}")

        # if self.verbose is True:
        #     print(f"{fp}:fp")
        # print(f"multisample_loc_file:{multisample_loc_file}")

        multisample_loc_file = multisample_loc_file if os.path.isfile(multisample_loc_file) else None

        return multisample_loc_file

    
    def _is_sample_computed(self, sample_fp:str):
        """ Returns whether the sample file was already computed. """
        class_name = self.__class__.__name__
        func_name = '_is_sample_computed'

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

    def _is_slide_computed(self, slide_fp:str):
        """ Returns whether the slide file was already computed. """


        class_name = self.__class__.__name__
        func_name = '_is_slide_computed'

        format = 'png' if (self.format == 'tiff' or self.format == 'tif') else 'txt'
        save_folder = os.path.join(self.save_root, 'images') if format == 'png' else os.path.join(self.save_root, 'labels')
        basename = os.path.split(slide_fp.split('.')[0])[1]
        txt_files = glob(os.path.join(self.folder, basename +'_sample*.txt'))
        

        # slide is computed if all samples are computed:
        computed = False
        for sample in txt_files:
            computed = True if self._is_sample_computed(sample_fp=sample) else computed

        if computed:
            self.log.warning(f"{class_name}.{func_name}: 😪😪 Tiler: found .{format} tiles in '{save_folder}' for {os.path.split(basename)[0]}.{format}. Skipping slide.")

        return computed
    

    # def _check_already_computed(self, fname: str, format: str, save_folder:str ):
    #     """ Checks if tiling is already computed for this WSI; if so, skips the slide. 
    #         Hypothesis: tiling is considered to be done if at least 2 tiles are found in 'save_folder'. """
    #     class_name = self.__class__.__name__
    #     func_name = '_check_already_computed'

    #     # # checking if multiple samples:
    #     # if self.multiple_samples is True:
    #     #     # get file with location of image/label samples within the slide:
    #     #     fp = os.path.join(self.folder, fname + f".{format}")
    #     #     multisample_loc_file = self._get_multisample_loc_file(fp, file_format='geojson')
    #     #     sample_locations = self._get_location_w_h(fp = multisample_loc_file) if multisample_loc_file is not None else [{'location':(0,0), 'w':W, 'h':H}]
    #     #     n_samples = len(sample_locations)

    #     # checking if tiles are already computed for each sample:
    #     format = 'png' if (format == 'tiff' or format == 'tif') else format
    #     files = glob(os.path.join(save_folder, f'*.{format}'))
    #     computed = True
    #     # for i in range(n_samples):
    #     name_like = fname + f"_sample{i}"
    #     matching_files = [file for file in files if name_like in file ]
    #     if format == 'png':
    #         computed = False if len(matching_files) <= 2 else computed
    #     else:
    #         computed = True if len(matching_files) >0  else computed

    #     if computed:
    #         self.log.warning(f"{class_name}.{func_name}: ❗️ Tiler: found .{format} tiles in '{save_folder}' for {fname}.{format}. Skipping slide.")

    #     return computed

    
    def _get_location_w_h(self, fp:str):
        """ Given a WSI with multiple samples and a file with annotations of the samples location 
            within the WSI, it returns the location, H, W for the openslide.read_region function to use. """
        

        assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")


        with open(fp, 'r') as f:
            data = geojson.load(f)
        all_dicts = []
        for rect in data['features']:

            assert len(rect['geometry']['coordinates'][0]) == 5, f"There seems to be more than 4 vertices annotated. "

            vertices = rect['geometry']['coordinates'][0][:-1]
            location = vertices[0]
            h =  vertices[1][1] - vertices[0][1]
            w =  vertices[2][0] - vertices[0][0]

            assert h>=0, f"{fp} has a feature with negative height. "
            assert w>=0, f"{fp} has a feature with negative width. "

            dictionary = {'location':location, 'w':w, 'h':h}
            all_dicts.append(dictionary)

    
        return all_dicts
    

    
    def test_show_image_labels(self, save = True):
        """ Shows K random images/labels. """
        class_name = self.__class__.__name__
        func_name = 'test_show_image_labels'
        

        @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Showing 6 random tiles:'" )
        def do(): 
            # pick random labels/images
            labels_fold = os.path.join(self.save_root, 'labels')
            labels = [os.path.join(labels_fold, file) for file in os.listdir(labels_fold)]
            labels = [file for file in labels if os.path.isfile(file)] # check labels exist
            images = [file.replace('labels', 'images').replace('.txt', '.png') for file in labels]
            pairs = list(zip(images, labels))
            pairs = [(img, lbl) for (img, lbl) in pairs if os.path.isfile(img) and os.path.isfile(lbl)]
            pairs = random.sample(pairs, k = 6)

            # show image + rectangles on labels:
            fig = plt.figure(figsize=(40,60))
            for i, (image_fp, label_fp) in enumerate(pairs):

                image = cv2.imread(image_fp)
                W, H = image.shape[:2]

                # read label
                with open(label_fp, 'r') as f:
                    text = f.readlines()
                    f.close()
                
                # draw rectangle for each glom/row:
                for row in text: 
                    items = row.split(sep = ' ')
                    xc, yc, box_w, box_h = [float(num) for num in items[1:]]
                    xc, box_w = xc * W, box_w * W
                    yc, box_h = yc * H, box_h * H
                    x0, x1 = int(xc - box_w / 2), int(xc + box_w / 2)
                    y0, y1 = int(yc - box_h/2), int(yc + box_h/2)
                    start_point = (x0, y0)
                    end_point = (x1,y1)
                    image = cv2.rectangle(img = image, pt1 = start_point, pt2 = end_point, color = (255,0,0), thickness=10)

                # add subplot with image
                plt.subplot(3,2,i+1)
                plt.tight_layout()
                plt.axis('off')
                plt.imshow(image)
            # title = [os.path.basename(img).replace('.png', '') for (img, _) in pairs ]
            # plt.suptitle(title)
            plt.show()

            fig.savefig(fname = os.path.join(self.save_root, 'show_tiles.png'))

                        

            return
        do()

        return
    
    def _get_sample_n(self, wsi_label_fp: str) -> int:
        """ Given teh label fp of the WSI, it returns the n of samples within it."""

        # 1) get the basename from txt file: 
        fn = os.path.basename(wsi_label_fp).split('.')[0]


        # 2) look matching tiles in the images folder and extract n_samples:
        image_dir = os.path.join(self.save_root, 'images')
        matching_tiles = [tile for tile in os.listdir(image_dir) if fn in tile]
        sample_numbers = [int(tile.split('sample')[-1][:1]) for tile in matching_tiles]
        sample_n = np.array(sample_numbers).max()

        # if self.verbose is True:
        #     print(f"n of tissue samples in the slide: {sample_n}")

        return sample_n

    # def _split_multisample_annotation(self, txt_file:str, multisample_loc_file:str) -> None:
    #     """ Given a WSI txt (not normalised) annotation, it splits the annotation file 
    #         into one file for each sample within the slide."""
        
    #     assert os.path.isfile(txt_file), f"'label_file':{txt_file} is not a valid filepath."
    #     assert os.path.isfile(multisample_loc_file), f"'label_file':{multisample_loc_file} is not a valid filepath."
    #     assert txt_file.split(".")[-1] == 'txt', f"'txt_file':{txt_file} should have '.txt' format. "

    #     with open(txt_file, 'r') as f:
    #         rows = f.readlines()
        
    #     with open(multisample_loc_file, 'r') as f:
    #         data = geojson.load(f)
        
    #     for row in rows:
    #         clss, xc, yc, box_w, box_h = row.replace(',', '').split(' ')
    #         clss, xc, yc, box_w, box_h = int(float(clss)), int(float(xc)), int(float(yc)), int(float(box_w)), int(float(box_h))
    #         for sample_n, rect in enumerate(data['features']):
    #             assert len(rect['geometry']['coordinates'][0]) == 5, f"There seems to be more than 4 vertices annotated. "
    #             vertices = rect['geometry']['coordinates'][0][:-1]
    #             x0, y0 = vertices[0]
    #             x1, y1 = vertices[2]


    #             if x0<xc<x1 and y0<yc<y1:
    #                 self.log.info(f"QUIIIIII{xc}-{x0}={xc - x0}")
    #                 text = f'{clss}, {xc - x0}, {yc - y0}, {box_w}, {box_h}\n'   # TODO DIVIDE TO NORMALIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                 # save txt file:
    #                 save_fp = txt_file.replace('.txt', f"_sample{sample_n}.txt")
    #                 with open(save_fp, 'a+') as f:
    #                     f.write(text)
        
    #     # return a list of txt files for each sample:
    #     txt_files = glob(os.path.join(self.folder, '*sample?.txt'))

    #     return txt_files


def test_Tiler():

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'



    print(" ########################    TEST 1: ⏳    ########################")
    folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\val\labels'
    save_root = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\val\labels'
    level = 2
    show = True
    target_format = 'txt'
    tiler = Tiler(folder = folder, 
                  tile_shape= (2048, 2048), 
                  step=512, 
                  save_root= save_root, 
                  level = level,
                  show = show,
                  verbose = True)
    # tiler(target_format='tif')
    if target_format == 'txt' and os.path.isdir('/Users/marco/Downloads/test_folders/test_tiler/labels'):
        fold = '/Users/marco/Downloads/test_folders/test_tiler/labels'
        files = [os.path.join(fold, file) for file in os.listdir(fold)]
        for file in tqdm(files, desc = 'Removing all label files'):
            os.remove(file)
    tiler(target_format=target_format)
    tiler.test_show_image_labels()

    print(" ########################    TEST 1: ✅    ########################")


    return


if __name__ == '__main__':
    test_Tiler()

    