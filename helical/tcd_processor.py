
# manager
import os,shutil
from typing import Literal
from tqdm import tqdm
from glob import glob
import numpy as np
from tiler import Tiler
from converter import Converter
from tcd_processor_base import ProcessorBase
import geojson
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import cv2
from loggers import get_logger
from utils import get_config_params
from cleaner import Cleaner


class Processor(ProcessorBase): 
    def __init__(self,
                 config_yaml_fp:str
                 ) -> None:
        
        super().__init__(config_yaml_fp=config_yaml_fp)
        self.config_yaml_fp = config_yaml_fp
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='processor')
        self.log = get_logger()

        self.map_classes = self.params['map_classes']
        self.stain = self.params['stain']
        self.resize = self.params['resize']
        self.multiple_samples = self.params['multiple_samples']
        self.class_name = self.__class__.__name__



        return
   
    
    def _rename_mrxsgson2gson(self)-> None:

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.mrxs.gson' in file]
        old_new_names = [(file, file.replace('.mrxs.gson', '.gson')) for file in files ]
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return
    

    def _convert_geojson2json(self)-> None: 
        """ TCD files should be 1) rect around sample in geojson fmt 2) slide annotation in .geojson BUT NAMED AS .json as to 
        not be confused with the rect geojson file 3) slide in .svs fmt. This func converts the NAMED .json file into a real .json fmt."""
        path_like = os.path.join(self.src_root, '*.json')
        json_files = glob(path_like)
        assert len(json_files)>0, f"'json_files' is empty. Path like: {path_like}."
        for file in json_files:
            with open(file=file, mode='r') as f: 
                data = geojson.load(f)
            data = data['features']
            with open(file=file, mode='w') as f: 
                geojson.dump(obj=data, fp=f)
        self.log.info(f"converted {file}")

        return
    

    def _make_geojson_files(self)-> None: 
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

        return
    

    def _tile_folder(self, dataset:Literal['train', 'val', 'test'])-> None:
        """ Tiles a single folder"""

        class_name = self.__class__.__name__
        func_name = self._tile_folder.__name__
        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        save_folder_labels = os.path.join(self.tiles_dir, dataset)
        save_folder_images = os.path.join(self.tiles_dir, dataset)

        # check that fold is not empty: 
        if len(os.listdir(slides_labels_folder)) == 0:
            self.log.warn(f"{class_name}.{func_name}: {dataset} fold is empty. Skipping.")
            return

        # 1) convert annotations to yolo format:
        if self.verbose_level in ['medium', 'high']: self.log.info(f"{class_name}.{func_name}: ⏳ 1) Converting annotations to YOLO format:")
        converter = Converter(config_yaml_fp=self.config_yaml_fp,
                                folder = slides_labels_folder) 
        converter()
        if self.verbose_level in ['medium', 'high']: self.log.info(f"{class_name}.{func_name}: ✅ 1) Converting annotations to YOLO format:")

        # 2) tile images:
        if self.verbose_level in ['medium', 'high']: self.log.info(f"{class_name}.{func_name}: ⏳ 2) Tiling slides to patches with shape {self.tiling_shape}:")
        tiler = Tiler(folder = slides_labels_folder, 
                      config_yaml_fp=self.config_yaml_fp,
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
        tiler(target_format=self.slide_format)
        if self.verbose_level in ['medium', 'high']: self.log.info(f"{class_name}.{func_name}: ✅ 2) Tiling slides to patches:")

        return
    

    def _segmentation2detection(self) ->None: 
        """ Converts segmentation YOLO .txt annotion files to bbox detection YOLO .txt files. """

        func_n = self._segmentation2detection.__name__

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
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                xc, yc = (x_min+x_max)/2, (y_min+y_max)/2
                w, h = x_max-x_min, y_max - y_min 
                new_text += str(class_n)
                new_text += f" {xc} {yc} {w} {h}"
                # new_text += f" {x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max}"
                new_text += '\n'
            
            with open(file, 'w') as f: # read file
                f.writelines(new_text)

        self.log.info(f"{self.class_name}.{func_n}: Converted mask segmentation annotations to bounding box labels for detection. ")

        return
    

    def _segmentation2detection_temptree(self) ->None: 
        """ Converts segmentation YOLO .txt annotion files to bbox detection YOLO .txt files. """

        func_n = self._segmentation2detection_temptree.__name__

        # get all label files:
        path_like = os.path.join(self.dst_root, self.task, 'temp', 'tiles', '*', 'labels', f"*.txt")
        self.format_msg(f"Path_like:{path_like}", func_n=func_n)
        labels = glob(path_like)
        labels = [file for file in labels if 'DS' not in file]
        assert len(labels)>0, f"'labels' like: {path_like} is empty."

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
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                xc, yc = (x_min+x_max)/2, (y_min+y_max)/2
                w, h = x_max-x_min, y_max - y_min 
                new_text += str(class_n)
                new_text += f" {xc} {yc} {w} {h}"
                # new_text += f" {x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max}"
                new_text += '\n'
            
            with open(file, 'w') as f: # read file
                f.writelines(new_text)

        self.log.info(f"{self.class_name}.{func_n}: Converted mask segmentation annotations to bounding box labels for detection. ")

        return
    

    def _rename_json_sample_pas(self)-> None:
        
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
    

    def _make_tif_files(self)-> None:
        """ Changes format from .png to .tif files. """
        images = glob(os.path.join(self.src_root, '*png'))
        images = [file for file in images if 'mask' not in file]

        for fp in images:
            dst = fp.replace('.png', '.tif')
            shutil.copy(src=fp, dst=dst)

        return
    

    def _del_regions_smaller_than_tile_size(self)-> None:
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
            except:
                delete=True
                self.log.error(f"Can't read {os.path.basename(geo_file)}. Deleting:")
            if delete is False:
                for j, sample in enumerate(data_rect['features']):
                    assert j==0, NotImplementedError("'_del_regions_smaller_than_tile_size' not tested for more than 1 sample")
                    rect_vertices = sample['geometry']['coordinates'][0]
                    assert len(rect_vertices) == 5, f"Geojson sample has {len(rect_vertices)} vertices, but should have 5."
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
        elif self.data_source == 'tcd':
            self._convert_geojson2json()
            
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
            self._del_regions_smaller_than_tile_size()
        # elif self.data_source == 'hubmap' or self.data_source == 'muw':
            # self._clean_muw_dataset()

        # clean and balance dataset
        cleaner = Cleaner(config_yaml_fp=self.config_yaml_fp)
        cleaner()

        if self.task == 'detection':
            self._segmentation2detection()
        if self.ignore_classes is not None:
            self._segmentation2detection_temptree()

        return


def test_processor():
    config_yaml_fp = '/Users/marco/yolo/code/helical/tcd_config_training.yaml'
    processor = Processor(config_yaml_fp=config_yaml_fp)
    processor._segmentation2detection()

    return
    



if __name__ == '__main__':
    config_yaml_fp = '/Users/marco/yolo/code/helical/config_mac_tcd.yaml'
    cleaner = Cleaner(config_yaml_fp=config_yaml_fp)
    cleaner()