# tiler 
import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from glob import glob
from patchify import patchify
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random
import json
import cv2
from tiler_base import TilerBase
from skimage import io



class Tiler(TilerBase): 

    def __init__(self, 
                map_classes: dict,
                *args,
                **kwargs):

        super().__init__(*args, **kwargs)
        self.map_classes = map_classes
        self.label_format = 'txt'
        self.slide_format = 'tif'
        self.tile_image_format = 'png'
        self.tile_label_format = 'txt'
        self._class_name = self.__class__.__name__
        # TODO TESTARE CHECK IS SLIDE COMPUTED 

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
       
        def do():  

            # if self._is_slide_computed(fp):
            #     return
            w, h = self.tile_shape
            # 1) read slide:
            if self.data_source == 'zaneta':
                # temporarely convert to png to 
                slide = cv2.imread(fp)
                slide = cv2.cvtColor(slide,cv2.COLOR_BGR2RGB)
                W, H = slide.shape[:2]
            else:
                try:
                    self.log.info(f"{class_name}.{func_name}: ⏳ Opening '{os.path.basename(fp)}':" )
                    slide = openslide.OpenSlide(fp)
                    W, H = slide.dimensions
                except:
                    self.log.error(f"{class_name}.{func_name}: ❌ Couldn t open file: '{os.path.basename(fp)}'. Skipping." )
                    return
            self.log.info(f"{class_name}.{func_name}: ✅ Opening '{os.path.basename(fp)}':" )
            self.log.info(f"SLIDE DIMENSIONSSS ARE:{W,H}")

            # 2) if file has multi_samples -> region = sample:
            # if self.multiple_samples is True:
                # get file with location of image/label samples within the slide:
            multisample_loc_file = self._get_multisample_loc_file(fp, file_format='geojson')
            sample_locations = self._get_location_w_h(fp = multisample_loc_file) if multisample_loc_file is not None else [{'location':(0,0), 'w':W, 'h':H}]
            # else:
            #     multisample_loc_file = None
            #     sample_locations = [{'location':(0,0), 'w':W, 'h':H}]

            for sample_n, sample in enumerate(sample_locations):

                location, W, H = sample['location'], sample['w'], sample['h']
                # self.log.info(f'location:{location}')
                # self.log.info(f'W:{W}')
                # self.log.info(f'H:{H}')

                # 1) reading region:
                self.log.info(f"{class_name}.{func_name}: ⏳ Reading region ({W, H}) of sample_{sample_n}:")
                if W<self.tile_shape[0] or H<self.tile_shape[1]:
                    self.log.info(f"Sample dimensions: {W, H}, tile shape: {self.tile_shape}. Region too small for the given tile shape.")
                    self.log.info(f"fileeeeeeee: {fp}")
                    continue
                
                if self.data_source == 'zaneta':
                    region = slide
                else:
                    try:
                        region = slide.read_region(location = location , level = self.level, size= (W,H)).convert("RGB")
                        self.log.info(f'Region sizeeeeeeeee {region.size}')
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
                patches = patchify(np_slide, (w, h, 3), step =  self.step )
                self.log.info(f"PATCH IMAGE DIMENSIONS: {patches.shape}")
                w_tiles,h_tiles = patches.shape[0],patches.shape[1]
                sample_fn = os.path.split(fp.replace('.tif', f"_sample{sample_n}"))[1]
                self._write_ntiles(sample_fn=sample_fn, dims=(w_tiles,h_tiles))

                # 3) save patches:
                self.log.info(f"{class_name}.{func_name}: ⏳ Saving patches of sample_{sample_n}:")
                patches = patches[:, :, 0, ...]
                fname = f"{os.path.split(fp)[1]}, sample {sample_n+1}/{len(sample_locations)}" if multisample_loc_file is not None else os.path.split(fp)[1]
                for i in tqdm(range(patches.shape[0]), desc= f"⏳ Tiling '{fname}'"):
                    for j in range(patches.shape[1]):
                        save_fp = fp.replace(f'.{self.format}', f'_sample{sample_n}_{i}_{j}.png') if self.multiple_samples else fp.replace(f'.{self.format}',f'_{i}_{j}.png')
                        if save_folder is not None:
                            fname = os.path.split(save_fp)[1]
                            save_fp = os.path.join(save_folder, fname)
                        else:
                            self.log.warn("save folder is none! Will overwrite previous ?")
                        
                        cv2_img = cv2.cvtColor(patches[i, j], cv2.COLOR_RGB2BGR)
                        cv2_img = cv2.resize(cv2_img, dsize=self.resize) # resize 
                        cv2.imwrite(save_fp, img = cv2_img)

                self.log.info(f"{class_name}.{func_name}: ✅ Saved sample_{sample_n} patches in {save_folder}.")

            # del patches
            return W, H
        
        W, H = do()
        return (W, H)
    

    def get_class_mask(self, json_file:str, region_dims:tuple) -> tuple: 
        """ Creates a mask which is all zeros except for vertices of gloms that are of value = glom unique id (increasing number)"""
        
        with open(json_file, mode='r') as f: 
            data = json.load(f)
        
        if len(data)==0: 
            self.log.warn('mask is empty. Skipping label.')
            return None, None, None


        # print(data)

        data = data['features']
        vertex_mask = np.zeros(shape=region_dims)
        class_mask = np.zeros_like(vertex_mask)
        order_mask = np.zeros_like(vertex_mask)
        zorro=0
        for i, glom in enumerate(data, start=1): 
            label_name = glom['classification']
            assert label_name in self.map_classes.keys(), self.log.error(f"{self.class_name}.{'read_label'}: class {label_name} not in 'map_classes': {self.map_classes}")
            label_val = self.map_classes[label_name] 
            vertices = glom['coordinates']


            assert all([len(vertex)==2 for vertex in vertices]), self.log.error(f"All vertices should be pairs of coordinates")
            
            for k,(x,y) in enumerate(vertices, start=1):
                if k==1:
                    zorro+=1
                x, y = min(x,region_dims[0]-1), min(y,region_dims[1]-1)
                x, y = int(x), int(y) # in hubmap annotations there's vertices like 3785.6, 23932.1 .. 
                vertex_mask[x,y] = i # assigning to each vertex a unique value (one for glom)
                class_mask[x,y] = label_val
                order_mask[x,y] = k
        
        # self._show_vertex_mask(vertex_mask=vertex_mask)
        
        self.log.info(f"Total gloms in this image: {i}")
        self.log.info(f"Total gloms assigned in this image: {zorro}")

        return  vertex_mask, class_mask, order_mask
    





    

    def _tile_class_mask(self, vertex_mask:np.ndarray, class_mask:np.ndarray,
                        order_mask:np.ndarray, save_folder:str, label_fp:str): 
        
        assert os.path.isdir(save_folder), self.log.error(f"{self.class_name}.{'_tile_class_mask'}: 'save_folder':{save_folder} is not a valid dirpath.")
        
        w, h = self.tile_shape
        W_mask, H_mask = vertex_mask.shape
        if W_mask<self.tile_shape[0] or H_mask<self.tile_shape[1]:
            self.log.info(f"Sample dimensions: {W_mask, H_mask}, tile shape: {self.tile_shape}. Region too small for the given tile shape.")
            return       

        # patchify masks:
        label_patches = patchify(vertex_mask, (w, h), step =  self.step )
        order_patches = patchify(order_mask, (w, h), step =  self.step )
        class_patches = patchify(class_mask, (w, h), step =  self.step )

        # self.log.info(f"PATCH LABEL DIMENSIONS: {label_patches.shape}")
        # self.log.info(f"PATCH ORDER DIMENSIONS: {order_patches.shape}")
        # self.log.info(f"PATCH CLASS DIMENSIONS: {class_patches.shape}")

        # loop through patches and write/save label_patch:
        for i in tqdm(range(label_patches.shape[0])):
            for j in range(label_patches.shape[1]):
                unique_values = np.unique(label_patches[i,j,:,:])
                unique_values = [val for val in unique_values if val != 0]
                if len(unique_values)==0: 
                    # self.log.info(f"In tile: _{j}_{i}.{self.tile_label_format} there seems to be no glom")
                    continue
                # self.log.info(f"yes ✅ there's label in tile: _{j}_{i}.{self.tile_label_format} ")

                # for each glom:
                text = ''
                for glom in unique_values: # each unique val corresponds to a glom
                    positions = np.argwhere(label_patches[i,j,:,:] == (glom)) # [(x3,y3), (x1,y1), (x5,y5)...]
                    # reorder positions based on order_patches:
                    order = [] # order of vertices is encoded in order_patches: in each vertex position (x,y) is stored the vertex order (to draw the polygon)
                    for x,y in positions:
                        order.append(order_patches[i,j,x,y] )
                    positions = [pos for (_, pos) in sorted(zip(order, positions), key=lambda tup: tup[0]) ]
                    # write text:
                    text += str(int(class_patches[i,j,x,y]))
                    for x_indices, y_indices in positions:
                        text+= f" {x_indices/self.tile_shape[0]} {y_indices/self.tile_shape[1]}"
                    text += '\n'
                # save in .txt file:
                if len(unique_values) > 0:
                    save_fn = os.path.basename(label_fp.split('.json')[0] + f'_{j}_{i}.{self.tile_label_format}')
                    replace_fold = lambda fp: os.path.join(  os.path.split(os.path.dirname(fp))[0]  , os.path.split(os.path.dirname(fp))[1].replace('images', 'labels'),     os.path.basename(fp))
                    save_fp = os.path.join(save_folder, save_fn)
                    save_fp = replace_fold(save_fp)
                    # self.log.info(f'save_fp:{save_fp}')
                    assert os.path.isdir(os.path.dirname(save_fp)), f"{os.path.dirname(save_fp)} is not a valid dirpath. Can't write a file here."
                    with open(save_fp, 'w') as f:
                        f.write(text)
  
        return
    
    
    def _get_tile_labels(self, fp: str, region_dims:tuple, save_folder: str = None):

        assert os.path.isfile(fp), self.log.error(ValueError(f"'fp':{fp} is not a valid filepath. "))
        assert 'json' in fp, self.log.error(ValueError(f"'fp':{fp} is not a json file. "))
        
        class_name = self.__class__.__name__
        func_name = '_get_tile_labels'
        save_folder = os.path.join(self.save_root, 'labels') if save_folder is None else save_folder

        def do():
            
            self.log.info(f"{class_name}.{'_get_tile_labels'}: Tiliing label '{fp}'") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            self.log.info(f"{class_name}.{'_get_tile_labels'}: creating class_mask:") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            vertex_mask, class_mask, order_mask = self.get_class_mask(json_file=fp, region_dims=region_dims)
            self.log.info(f"{class_name}.{'_get_tile_labels'}: patchifying class_mask:") # D:\marco\datasets\muw_retiled\wsi\test\labels\200701099_09_SFOG_sample0.json
            if vertex_mask is not None:
                self._tile_class_mask(vertex_mask=vertex_mask, class_mask=class_mask, 
                                    order_mask=order_mask, save_folder=save_folder, label_fp=fp)
           
            return
        do()
        
        return  


    def __call__(self, target_format: str, save_folder: str = None) -> None:
        """ Tiles/patchifies WSI or annotations. """

        SLIDE_FORMATS =  ['tif']
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
        assert len(fnames) > 0, f"'fnames' is empty."

        # 3) tile files:
        if len(files) == 0: 
            self.log.error(f"{class_name}.{func_name}: ❌ No file in format '{target_format}' was found in '{self.folder}'.")

        # check if all tiling is complete:
        _, uncompleted = self._get_completed_files(fnames)
        if len(uncompleted) == 0: 
            self.log.info(f"{class_name}.{func_name}: 🎉 All files in folder '{os.path.dirname(files[0])}' have been tiled and saved in '{save_folder}'.  ")
            self.show = True


        for i, file in enumerate(files):
            print(f'####### PROCESSING SLIDE {i}/{len(files)} #######')
            if self.multiple_samples:
                json_samples = glob(os.path.join(os.path.dirname(file), f"*{os.path.basename(file).split('.')[0]}*.json"))
                json_samples = [file for file in json_samples if "sample" in file]
                # print(f'file:{file}')
                assert len(json_samples)>0, f"'json_samples' like :{os.path.join(os.path.dirname(file), f'*{file}*.json')} is empty."
                # self.log.info()
                    
                W, H = self._get_tile_images(fp = file, save_folder=save_folder )
                self.log.info(f"Tiling labels using {W,H}")
                for json_sample in json_samples:
                    self._get_tile_labels(fp = json_sample, region_dims=(W,H), save_folder=save_folder)

            # if self.show is True:
            #     self.test_show_image_labels()

        return
    
    
    def test_show_image_labels(self, save = True):
        """ Shows K random images/labels. """

        replace_dir = lambda fp, to_dir, format: os.path.join(os.path.dirname(os.path.dirname(fp)), to_dir, os.path.basename(fp).split('.')[0] + f".{format}")
        labels = glob(os.path.join(self.save_root, 'labels', '*.txt')) # /Users/marco/helical_tests/test_manager_detect_muw_sfog/detection/tiles/train
        k=min(4, len(labels))

        # 1) Picking images:
        labels = random.sample(labels, k=k)
        pairs = [(replace_dir(fp, to_dir='images', format='png'), fp) for fp in labels]
        pairs = list(filter(lambda pair: (os.path.isfile(pair[0]) and os.path.isfile(pair[1])), pairs))
        # self.log.info(f"Displaying {[os.path.basename(label) for label in labels]}")
        # 2) Show image/drawing rectangles as annotations:
        fig = plt.figure(figsize=(20, k//2*10))
        for i, (image_fp, label_fp) in enumerate(pairs):

            # read image
            image = cv2.imread(image_fp)
            W, H = image.shape[:2]
            # print((W, H))
            # self.log.info(f"image shape: {W,H}")

            # read label
            with open(label_fp, 'r') as f:
                text = f.readlines()
                f.close()
            # draw rectangle for each glom/row:
            for row in text: 
                row = row.replace('/n', '')
                items = row.split(sep = ' ')
                class_n = int(float(items[0]))
                items = items[1:]
                x = [el for (j,el) in enumerate(items) if j%2 == 0]
                x = [np.int32(float(el)*W) for el in x]
                y = [el for (j,el) in enumerate(items) if j%2 != 0]
                y = [np.int32(float(el)*H) for el in y]
                vertices = list(zip(x,y)) 
                vertices = [list(pair) for pair in vertices]
                vertices = np.array(vertices, np.int32)
                vertices = vertices.reshape((-1,1,2))
                x0 = np.array(x).min()
                y0 = np.array(y).min()  
                if self.data_source == 'zaneta':                  
                    color = (0,255,0) if class_n == 1 else (255,0,0) 
                elif self.data_source == 'hubmap':
                    # assert class_n == 0, f"Class to display is not 0, but hubmap should only contain class 0 objects."
                    color = (0,255,0) if class_n == 0 else (255,0,0) 
                elif self.data_source == 'muw':
                    color = (0,255,0) if class_n == 0 else (255,0,0) 
                image = cv2.fillPoly(image, pts = [vertices], color=color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = 'H' if class_n == 0 else 'U'
                image = cv2.putText(image, text, org = (x0,y0-H//50), color=color, thickness=3, fontFace=font, fontScale=1)

            # add subplot with image
            image = cv2.addWeighted(image, 0.4, cv2.imread(image_fp), 0.6, 1.0)
            plt.subplot(k//2,2,i+1)
            plt.title(f"Example tile.")
            plt.imshow(image)
            plt.tight_layout()
            plt.axis('off')
        
        plt.show()
        fig.savefig('tiled_examples.png')

        return


    def _is_slide_computed(self, sample_fp:str):
        """ Returns whether the sample file was already computed. """
        
        class_name = self.__class__.__name__
        func_name = '_is_slide_computed'

        format = 'png' if (self.format == 'tiff' or self.format == 'tif') else 'txt'
        save_folder = os.path.join(self.save_root, 'images') if format == 'png' else os.path.join(self.save_root,'labels')
        basename = os.path.split(sample_fp.split('.')[0])[1] # e.g. 200104066_09_SFOG_sample0
        name_like = os.path.join(save_folder, basename) # e.g. /Users/marco/Downloads/test_folders/test_tiler/test_1slide/images/200104066_09_SFOG_sample0
        name_like = f'{name_like}*.{format}' # e.g. /Users/marco/Downloads/test_folders/test_tiler/test_1slide/images/200104066_09_SFOG_sample0*.png
        matching_files = glob(name_like)
        computed = False if len(matching_files) <= 8 else True

        if computed:
            self.log.warning(f"{class_name}.{func_name}: 😪 Tiler: found .{format} tiles in '{save_folder}' for {os.path.split(basename)[0]}.{format}. Skipping sample.")

        return computed