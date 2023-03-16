
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



class TilerSegm(Tiler): 

    def __init__(self, 
                map_classes: dict = {'Glo-unhealthy':0, 'Glo-NA':1, 'Glo-healthy':2, 'Tissue':3},
                inflate_points_ntimes:int = None,
                *args,
                **kwargs):

        super().__init__(*args, **kwargs)
        self.map_classes = map_classes
        self.label_format = 'json'
        self.slide_format = 'tif'
        self.tile_image_format = 'png'
        self.tile_label_format = 'txt'
        self.inflate_points_ntimes = inflate_points_ntimes

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
            self.log.info(f"SLIDE DIMENSIONSSS ARE:{W,H}")

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
                    self.log.info(f"image {fp} patches shape: {patches.shape}")
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

            return W, H
        
        W, H = do()
        # self.log.info(f"returninnnnng {W,H}")
        return (W, H)
    

    def get_class_mask(self, json_file:str, region_dims:tuple) -> tuple: 
        """ Creates a mask which is all zeros except for vertices of gloms that are of value = glom unique id (increasing number)"""
        
        # self.log.info('data reading')
        with open(json_file, mode='r') as f: 
            data = json.load(f)
        
        # self.log.info('data read')
        vertex_mask = np.zeros(shape=region_dims)
        # self.log.info(f"created array of shape {region_dims}")
        class_mask = np.zeros_like(vertex_mask)
        order_mask = np.zeros_like(vertex_mask)
        # self.log.info('np arrays created')
        for i, glom in enumerate(data):
            label_name = glom['properties']['classification']['name']
            # self.log.info('got label name')
            assert label_name in self.map_classes.keys(), self.log.error(f"{self.class_name}.{'read_label'}: class {label_name} not in 'map_classes': {self.map_classes}")
            # self.log.info('assert passed')
            label_val = self.map_classes[label_name]
            # self.log.info('got label val')
            vertices = glom['geometry']['coordinates'][0]
            # self.log.info('got vertices')

            # self.log.info(f"vertices pre: {len(vertices)}")
            # self.log.info(f"vertices pre: {vertices[0]}")

            def _inflate(old_vertices:list):
                # with inflation:
                new_vertices = []
                assert len(old_vertices)>=3, self.log.error(f"{self.class_name}.{'_inflate'}: label {json_file} has a glom with only {len(old_vertices)} vertices")
                for i in range(len(old_vertices)-1):
                    # self.log.info(f"vertex: {vertices[i]}")
                    x_i,y_i = (old_vertices[i])
                    # self.log.info(x_i,y_i)

                    x_l,y_l = (old_vertices[i+1])
                    x_middle = (x_l + x_i)/2
                    # self.log.info(f"x_middle: {x_middle}")
                    y_middle = (y_l+ y_i)/2
                    new_vertices.extend([[x_i, y_i], [x_middle, y_middle]])
                    # new_row += f" {x[i]} {y[i]} {x_middle} {y_middle} {x[i+1]} {y[i+1]}"
                
                return new_vertices
            
            # for i in range()
            
            if self.inflate_points_ntimes is not None:
                for i in range(self.inflate_points_ntimes):
                    self.log.info(f"inflation number {i}. Vertices: {len(vertices)}")
                    vertices = _inflate(old_vertices=vertices)
            # self.log.info(f"vertices post: {len(vertices)}")
            # self.log.info(f"vertices post: {vertices[0]}")


            # without inflation
            assert len(vertices)%2 ==0, self.log.error(f"Vertices should be even, but are {len(vertices)}")
            for k,(x,y) in enumerate(vertices):
                # self.log.info(f'got x:{x}, got y:{y}')
                x, y = int(x), int(y) # slice values must be int 
                vertex_mask[x,y] = i # assigning to each vertex a unique value (one for glom)

                # self.log.info('assigned val to vertex')
                class_mask[x,y] = label_val
                order_mask[x,y] = k
            
                # self.log.info(f"adding {k}")
            # self.log.info(f"{k} verteces in this glom")
        # self.log.info(f" done with the mask. Assigned {i} unique values to this mask")

        return  vertex_mask, class_mask, order_mask
    

    # def inflate_annotation_points(self, label_fp:str) -> None: 
    #     """ Interpolates between annotation points so as to inflate the number of points for a given label file. """

    #     assert os.path.isfile(label_fp), self.log.error(f"{self.class_name}.{'inflate_annotation_points'}: 'label_fp':{label_fp} is not a valid filepath.")

    #     self.log.info(f"{self.class_name}.{'inflate_annotation_points'}: Inflating:")

    #     with open(label_fp, 'r') as f:
    #         text = f.readlines()
    #     # self.log.info("read text file inflation")
        
    #     new_rows = []
    #     for row in text: 
    #         row = row.replace('\n','')
    #         items = row.split(sep = ' ')
    #         class_n = items[0]
    #         items = items[1:]
    #         x = [float(el) for (i, el) in enumerate(items) if i%2==0] # even idcs = x coords
    #         y = [float(el) for (i, el) in enumerate(items) if i%2!=0] # uneven idcs = y coords
    #         # self.log.info(f"got x y from text: x:{len(x)}, y:{len(y)}")

    #         new_row = f"{class_n}"
    #         for i in range(len(x)-1): 
    #             x_middle = (x[i+1] + x[i])/2
    #             # self.log.info(f"x_middle: {x_middle}")
    #             y_middle = (y[i+1] + y[i])/2
    #             new_row += f" {x[i]} {y[i]} {x_middle} {y_middle} {x[i+1]} {y[i+1]}"
    #             # self.log.info(f"new_row: {new_row}")
    #         # self.log.info(f"changed row")
    #         # new_row+= '\n'
    #         new_rows.append(new_row)

    #     # self.log.info(f"writing file")
    #     with open(label_fp, 'w') as f:
    #         f.writelines(new_rows)
    #     # self.log.info(f"File written ")
        
    #     return
    



    


    def _tile_class_mask(self, vertex_mask:np.ndarray, class_mask:np.ndarray,
                        order_mask:np.ndarray, save_folder:str, label_fp:str): 
        
        # self.log.info(f"tile class")
        assert os.path.isdir(save_folder), self.log.error(f"{self.class_name}.{'_tile_class_mask'}: 'save_folder':{save_folder} is not a valid dirpath.")
        # self.log.info(f"tile class2")
        w, h = self.tile_shape

        label_patches = patchify(vertex_mask, (w, h), step =  self.step )
        order_patches = patchify(order_mask, (w, h), step =  self.step )
        # self.log.info(f"tile class3")
        # self.log.info(f"label {label_fp} patches shape: {label_patches.shape}")
        # self.log.info(f"{self.class_name}.{'_tile_class_mask'}: patches shape:{label_patches.shape}")
        for i in tqdm(range(label_patches.shape[0])):
            for j in range(label_patches.shape[1]):
                # self.log.info(f"unique class2")
                unique_values = np.unique(label_patches[i,j,:,:])
                unique_values = [val for val in unique_values if val != 0]
                # self.log.info(f"unique_values for this tile:{unique_values}, type: {type(unique_values)}")
                
                if len(unique_values)==0: 
                    continue
                # self.log.info('boh')
                text = ''
                # for each glom:
                for glom in unique_values: # each unique val corresponds to a glom
                    # self.log.info(f"glom: {glom}")

                    # self.log.info(f"label patches[i,j,:,:] shape: {label_patches[i,j,:,:].shape}")
                    # self.log.info(f"order patches: {order_patches[i,j,:,:]}")
                    # self.log.info(f"shape: {order_patches[i,j,:,:].shape}")
                    # self.log.info(f"type: {type(order_patches[i,j,:,:])}")
                    positions = np.argwhere(label_patches[i,j,:,:] == (glom)) # [(x3,y3), (x1,y1), (x5,y5)...]
                    # self.log.info(f"positions pre {positions}")
                    # [
                    #       1
                    #   0       2
                    #      3     ]
                    # [
                    #       0
                    #   0       0
                    #      0    ]
                    order = []
                    self.log.info(f"got pre loop")
                    for x,y in positions:
                        # self.log.info(f"xy: {x,y}")
                        order.append(order_patches[i,j,x,y] ) # e.g. 5th -> [2nd, 5th, 8th..]
                        # self.log.info(f"order: {order}")
                        # self.log.info(f"appending")
                    self.log.info(f"got out loop")
                    # self.log.info(order)
                    # self.log.info(positions)
                    # self.log.info(len(order))
                    self.log.info(len(positions))

                    positions = [pos for (_, pos) in sorted(zip(order, positions)) ]
                    self.log.info(f"positions post {positions}")
                    # self.log.info(f"reordered")

                    # self.log.info(f"positions:{positions}")


                    text += "0"
                    for x_indices, y_indices in positions:
                        # self.log.info(f"x_indices: {x_indices}")
                        # self.log.info(f"y_indices: {y_indices}")

                        text+= f" {x_indices} {y_indices}"
                    self.log.info(f"added  'x_indices' 'y_indices'")
                    text += '\n'
                    
                
                # save in .txt file:
                if len(unique_values) > 0:
                    # self.log.info(f"text: {text}")
                    save_fn = os.path.basename(label_fp.replace(f".{self.label_format}",f'_{j}_{i}.{self.tile_label_format}'))
                    replace_fold = lambda fp: os.path.join(os.path.dirname(fp).replace('images', 'labels'), os.path.basename(fp))
                    save_fp = os.path.join(save_folder, save_fn)
                    # self.log.info(f"save_fp before: {save_fp}")
                    save_fp = replace_fold(save_fp)
                    # self.log.info(f"save_fp : {save_fp}")
                    with open(save_fp, 'w') as f:
                        f.write(text)
                    # self.log.info(f"inflating:")
                    # self.inflate_annotation_points(save_fp)
        return
    
    
    def _write_tiled_label(self, text:str, label_fp:str, save_folder:str):

        #### CONTINUEEEEEEEEE HEREEEEEEEE ######


        

        return
    

    def _get_tile_labels(self, fp: str, region_dims:tuple, save_folder: str = None):

        assert os.path.isfile(fp), self.log.error(ValueError(f"'fp':{fp} is not a valid filepath. "))
        assert 'json' in fp, self.log.error(ValueError(f"'fp':{fp} is not a json file. "))


        
        class_name = self.__class__.__name__
        func_name = '_get_tile_labels'
        save_folder = os.path.join(self.save_root, 'labels') if save_folder is None else save_folder

        # @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Tiling label: '{os.path.basename(fp)}'" )
        def do():
            
            self.log.info(f"{class_name}.{'_get_tile_labels'}: Tiliing label '{fp}'") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            self.log.info(f"{class_name}.{'_get_tile_labels'}: creating class_mask:") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            vertex_mask, class_mask, order_mask = self.get_class_mask(json_file=fp, region_dims=region_dims)
            self.log.info(f"{class_name}.{'_get_tile_labels'}: patchifying class_mask:") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            self._tile_class_mask(vertex_mask=vertex_mask, class_mask=class_mask, order_mask=order_mask, save_folder=save_folder, label_fp=fp)
            

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
            # self._remove_small_files(files=files)
            W, H = self._get_tile_images(fp = file, save_folder=save_folder )
            self._get_tile_labels(fp = file.replace('.tif', '.json'), region_dims=(W,H), save_folder=save_folder )

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