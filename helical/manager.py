# manager
import os,shutil
from typing import Literal
from tqdm import tqdm
from glob import glob
import numpy as np
from tiler import Tiler
from converter import Converter
from manager_base import ManagerBase
import geojson
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
# from skimage import io
import cv2



class Manager(ManagerBase): 

    def __init__(self,
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        # assert self.data_source == 'muw', self.log.error(ValueError(f"'data_source' is {self.data_source} but Manager used is 'ManagerMUW'"))
        # self.data_source == "muw"

        return

    
    def _rename_mrxsgson2gson(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.mrxs.gson' in file]
        old_new_names = [(file, file.replace('.mrxs.gson', '.gson')) for file in files ]
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return
    

    def _make_geojson_files(self) -> None: 
        """ For HubMAP samples/slides, it creates a geojson sample around all the image. """
        
        # get slide files:
        slides_fp = glob(os.path.join(self.src_root, '*.tif' ))
        dictionary = {"type":"FeatureCollection",
                        "features":[{"type":"Feature",
                                    "geometry":{"type":"Polygon",
                                                "coordinates":[[[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]]},
                                    "properties":{"object_type":"annotation","isLocked":False}}] }

        for slide_fp in tqdm(slides_fp, desc='making geojson files'): 
            # get bbox around the whole sample/slide
            if self.data_source == 'hubmap':
                slide = openslide.OpenSlide(slide_fp) # open slide
                w, h = slide.dimensions # i.e. max coords
            elif self.data_source == 'zaneta': 
                slide = cv2.imread(slide_fp)
                slide = cv2.cvtColor(slide,cv2.COLOR_BGR2RGB)
                w,h = slide.shape[:2]
            coordinates = [[0, 0],[0, h],[w, h],[w, 0],[0, 0]]
            dictionary['features'][0]['geometry']['coordinates'][0] = coordinates
            geojson_file = slide_fp.replace('.tif', '_PAS.geojson')
            # write geojson file
            with open(geojson_file, 'w') as fs:
                geojson.dump(obj = dictionary, fp = fs)
        
        self.log.info(f'making geojson file using dimensions.')

        # print('dajeeee')
        # raise NotImplementedError()

        return



    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        """ Tiles a single folder"""

        class_name = self.__class__.__name__
        func_name = '_tile_folder'
        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        save_folder_labels = os.path.join(self.tiles_dir, dataset)
        save_folder_images = os.path.join(self.tiles_dir, dataset)

        # check that fold is not empty: 
        if len(os.listdir(slides_labels_folder)) == 0:
            self.log.warn(f"{class_name}.{func_name}: {dataset} fold is empty. Skipping.")
            return

        # 1) convert annotations to yolo format:
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ⏳    ########################")
        # if self.data_source == 'muw':
        converter = Converter(folder = slides_labels_folder, 
                              map_classes = self.map_classes,
                                stain = self.stain, 
                                data_source = self.data_source,
                                multiple_samples = self.multiple_samples,
                                convert_from='gson_wsi_mask',  
                                convert_to='txt_wsi_bboxes',
                                save_folder= slides_labels_folder, 
                                level = self.tiling_level,
                                verbose=self.verbose)
        converter()
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ✅    ########################")

        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        tiler = Tiler(folder = slides_labels_folder, 
                      map_classes=self.map_classes,
                    tile_shape= self.tiling_shape, 
                    step=self.tiling_step, 
                    data_source = self.data_source,
                    save_root= save_folder_images, 
                    level = self.tiling_level,
                    show = self.tiling_show,
                    verbose = self.verbose,
                    resize = self.resize,
                    multiple_samples = self.multiple_samples)

        target_format = 'tif'
        tiler(target_format=target_format)
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

        return
    

    def _segmentation2detection(self): 

        # get all label files:
        labels = glob(os.path.join(self.dst_root, self.task, 'tiles', '*', 'labels', f"*.txt"))
        labels = [file for file in labels if 'DS' not in file]
        assert len(labels)>0, f"'labels' like: {os.path.join(self.dst_root, self.task, 'tiles', '*', 'labels',  f'*.txt')} is empty."

        # loop through labels and get bboxes:
        for file in tqdm(labels, desc='transforming segm labels to bboxes'): 
            with open(file, 'r') as f: # read file
                text = f.readlines()
            new_text=''
            for row in text: # each row = glom vertices
                row = row.replace(' /n', '')
                items = row.split(sep = ' ')
                class_n = int(float(items[0]))
                items = items[1:]
                x = [el for (j,el) in enumerate(items) if j%2 == 0]
                x = np.array([float(el) for el in x])
                y = [el for (j,el) in enumerate(items) if j%2 != 0]
                y = np.array([float(el) for el in y])
                x_min, x_max = str(x.min()), str(x.max())
                y_min, y_max = str(y.min()), str(y.max())
                new_text += str(class_n)
                new_text += f" {x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max}"
                new_text += '\n'
            
            with open(file, 'w') as f: # read file
                f.writelines(new_text)

        # use self.tiler to show new images:
        print('plotting images')
        # self.tiler.test_show_image_labels()
        print('plotting images done. ')

        return
    
    def _rename_json_sample_pas(self):
        
        # rename geojson, add _PAS_sample0:
        slides_fp = glob(os.path.join(self.src_root, '*.json' ))
        for fp in slides_fp: 
            dst = os.path.join(os.path.dirname(fp), os.path.basename(fp).replace('.json','_PAS.json'))
            if not os.path.isfile(dst):
                os.rename(src=fp, dst=dst)

        # rename slides:
        images_fp = glob(os.path.join(self.src_root, '*.tif' ))
        for fp in images_fp: 
            dst = os.path.join(os.path.dirname(fp), os.path.basename(fp).replace('.tif','_PAS.tif'))
            if not os.path.isfile(dst):
                os.rename(src=fp, dst=dst)

        return
    
    def _make_tif_files(self):
        images = glob(os.path.join(self.src_root, '*png'))
        images = [file for file in images if 'mask' not in file]

        for fp in images:
            dst = fp.replace('.png', '.tif')
            shutil.copy(src=fp, dst=dst)



        return
    

    def _del_regions_smaller_than_tile_size(self):
        # /Users/marco/Downloads/zaneta_files/detection copia/wsi/train/labels
        # /Users/marco/Downloads/zaneta_files/detection/wsi/test/labels/I_4_S_5_ROI_4_PAS.geojson

        # TODO FUNZIONA SOLO SU ZANETA, PERCHE' SE CI SONO VARI SAMPLE NON VA
        path_like = os.path.join(self.dst_root, self.task, 'wsi', '*', 'labels', '*.geojson' )
        geojson_files = glob(path_like)
        geojson_files = [file for file in geojson_files if os.path.isfile(file)]

        for geo_file in geojson_files:
            self.log.info(f'geojson: {geo_file}')
            delete = False
            try:
                with open(geo_file, 'r') as f:
                    data_rect = geojson.load(f)
                # self.log.info(f"Could successfully read {os.path.basename(geo_file)}. yeah:")
            except:
                delete=True
                self.log.error(f"Can't read {os.path.basename(geo_file)}. Deleting:")
            if delete is False:
                for j, sample in enumerate(data_rect['features']):
                    assert j==0, NotImplementedError("'_del_regions_smaller_than_tile_size' not tested for more than 1 sample")
                    rect_vertices = sample['geometry']['coordinates'][0]
                    assert len(rect_vertices) == 5, f"Geojson sample has {len(rect_vertices)} vertices, but should have 5."
                    # x_start,y_start = rect_vertices[0]
                    sample_w, sample_h = rect_vertices[2]
                    if self.tiling_shape[0]>sample_w or self.tiling_shape[1]>sample_h:
                        self.log.error(f"Sample size: {sample_w, sample_h}. Tile size: {self.tiling_shape}")
                        delete=True


            if delete is True:
                sample_file = os.path.join(os.path.dirname(geo_file), os.path.basename(geo_file).replace('.geojson', 'json'))
                json_sample = os.path.join(os.path.dirname(geo_file), os.path.basename(geo_file).replace('.geojson', '_sample0.json'))
                del_files = [geo_file, sample_file, json_sample]
                del_files = [file for file in del_files if os.path.isfile(file)]
                for del_file in del_files: 
                    os.remove(del_file)
                self.log.info(f"Removed {os.path.basename(geo_file).split('.')[0]}. Region smaller than tile_size ")


        return


    def __call__(self) -> None:
        
        self._rename_tiff2tif()

        if self.data_source == 'hubmap':
            self._make_geojson_files()
            self._rename_json_sample_pas()
        elif self.data_source == 'muw':
            self._rename_mrxsgson2gson()
        elif self.data_source == 'zaneta':
            self._make_tif_files()
            self._make_geojson_files()
            self._rename_json_sample_pas()
        
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

        if self.data_source == 'zaneta':
            # raise NotImplementedError()
            self._del_regions_smaller_than_tile_size()
        else:
            self._clean_muw_dataset()

        # self._clean_muw_dataset()

        if self.task == 'detection':
            self._segmentation2detection()
            self.log.info('segm 2 detection done')

        return
    



