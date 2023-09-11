import os 
from glob import glob
from typing import List
from configurator import Configurator
from abc import ABC, abstractmethod
from utils import get_config_params
from tqdm import tqdm
import pandas as pd
import numpy as np


class Profiler(Configurator, ABC): 

    def __init__(self, 
                 config_yaml_fp:str)->None:
                # data_root:str, 
                # wsi_images_like:str,
                # wsi_labels_like:str,
                # tile_images_like:str,
                # tile_labels_like:str,
                # empty_ok:bool = False,
                # verbose:bool=False,
                # skip_test:bool = False) -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            wsi_image_like= e.g. '*.tif'
            wsi_label_like = e.g. '*_sample?.txt'
            tile_image_like = e.g. '*sample*.png'
            tile_label_like = e.g. '*sample*.txt'
            '"""
        super().__init__()
        self.params = get_config_params(yaml_fp=config_yaml_fp, config_name='processor')
        self._set_all_attrs()
        self._parse_args()

        return
    
    def _set_all_attrs(self)->None:
        """ Sets all class attributes. """
        func_n = self._set_all_attrs.__name__

        self._class_name = self.__class__.__name__
        self.data_root = os.path.join(self.params['dst_root'], self.params['task'])
        self.wsi_image_format = self.params['slide_format']
        self.wsi_label_format = self.params['label_format']
        self.wsi_images_like = f"*.{self.wsi_image_format}"
        self.wsi_labels_like = f"*_sample?.{self.wsi_label_format}"
        self.tile_images_like = '*sample*.png'
        self.tile_labels_like = '*sample*.txt'
        self.tiles_image_format = 'png'
        self.tiles_label_format = 'txt'
        self.verbose = self.params['verbose']
        self.skip_test = True if self.params['crossvalidation'] is True else False
        self.already_written = False

        return
    
    def _parse_args(self)->None:
        assert os.path.isdir(self.data_root), f"'data_root':{self.data_root} is not a valid dirpath."

        return

    
    def _get_data(self) -> dict:
        """ From a roots like root -> wsi/tiles->train,val,test->images/labels, 
            it returns a list of wsi images/labels and tiles images/labels."""
        
        # look for files;
        wsi_images = glob(os.path.join(self.data_root, 'wsi', '*', 'images', self.wsi_images_like))
        wsi_labels = glob(os.path.join(self.data_root, 'wsi', '*', 'labels', self.wsi_labels_like))
        tile_images = glob(os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like))
        tile_labels = glob(os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like))

        # in case of e.g. data cleaning, test files are NOT to be changed.
        if self.already_written is False:
            self.log.info(f"{self._class_name}.{'_get_data'}: Ignoring test files in cleaning dataset.")
        is_not_test = lambda path: os.path.basename(os.path.dirname(os.path.dirname(path))) != 'test'
        wsi_images = list(filter(is_not_test, wsi_images))
        wsi_labels = list(filter(is_not_test, wsi_labels))
        tile_images = list(filter(is_not_test, tile_images))
        tile_labels = list(filter(is_not_test, tile_labels))
   
        data = {'wsi_images':wsi_images, 'wsi_labels':wsi_labels, 'tile_images':tile_images,  'tile_labels':tile_labels }
        
        # if allowed to be empty:
        # if self.empty_ok is False:
        #     assert len(tile_images) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
        #     assert len(tile_labels) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
        # else:
        if len(tile_images) == 0: 
            self.log.warning(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
        if len(tile_labels) == 0:
            self.log.warning(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
        # if len(wsiself.log.info(f"{self._class_name}.{'_get_data'}: Found {len(wsi_images)} slides with {len(wsi_labels)} annotations and {len(wsi_images)} tiles with {len(wsi_labels)} annotations. ")
        self.already_written = True

        return data
    

    def _get_unique_labels(self, verbose = False) -> dict:
        """ Returns unique label values from the dataset. """

        # look thorugh labels:
        unique_labels = []
        for label_fp in self.data['tile_labels']:
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                labels = [row[0] for row in rows]
                unique_labels.extend(labels)
        # get unique vals:
        unique_labels = list(set(unique_labels))
        
        if verbose is True:
            self.log.info(f"{self._class_name}.{'_get_unique_labels'}: Unique classes: {unique_labels}", )

        return unique_labels
        
    
    def _get_class_freq(self) -> dict:
        """ Gets classes and their frequency in the dataset."""

        assert hasattr(self, 'n_classes'), self.log.error(AttributeError(f"{self._class_name}.{'_get_class_freq'}: object doesn't have a 'n_classes' attribute."))

        class_freq = dict(zip([str(num) for num in range(self.n_classes)], [0 for _ in range(self.n_classes)]))
        for label_fp in self.data['tile_labels']:
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                labels = [row[0] for row in rows]
                for label in labels:
                    class_freq[label] += 1

        if self.verbose is True:
            self.log.info(f"{self._class_name}.{'_get_class_freq'}: class_freq: {class_freq}")

        return class_freq
    
    def _get_only_train_val_files(self):

            wsi_images = self.data['wsi_images']
            wsi_labels = self.data['wsi_labels']
            tile_images = self.data['tile_images']
            tile_labels = self.data['tile_labels']
            
            is_not_test = lambda path: os.path.basename(os.path.dirname(os.path.dirname(path))) != 'test'
            wsi_images = list(filter(is_not_test, wsi_images))
            wsi_labels = list(filter(is_not_test, wsi_labels))
            tile_images = list(filter(is_not_test, tile_images))
            tile_labels = list(filter(is_not_test, tile_labels))

            filtered_data = {'wsi_images':wsi_images, 'wsi_labels':wsi_labels, 
                            'tile_images':tile_images, 'tile_labels':tile_labels}

            return filtered_data
    
    def _get_empty_images(self, also_from_test: bool = True):
        """ Returns full and empty images from the dataset. """

        class_name = self.__class__.__name__
        func_name = '_get_empty_images'

        if also_from_test is False:
            filtered_data = self._get_only_train_val_files()
            tile_images = filtered_data['tile_images']
            tile_labels = filtered_data['tile_labels']
        else:
            tile_images = self.data['tile_images'] 
            tile_labels = self.data['tile_labels'] 

        assert len(tile_images)>0, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'tile_images':{len(tile_images)}. No tile image found. ")
        assert len(tile_labels)>0, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'tile_labels':{len(tile_labels)}. No tile label found. ")
        
        rename_img2lbl = lambda fp_img: os.path.join(os.path.dirname(fp_img).replace('images', 'labels'), os.path.basename(fp_img).replace(self.tiles_image_format, self.tiles_label_format))
        rename_lbl2img = lambda fp_lbl: os.path.join(os.path.dirname(fp_lbl).replace('labels', 'images'), os.path.basename(fp_lbl).replace(self.tiles_label_format,self.tiles_image_format))
        
        empty_images = [file for file in tile_images if rename_img2lbl(file) not in tile_labels]
        empty_images = [file for file in empty_images if "DS" not in empty_images and self.tiles_image_format in file]
        empty_images = [file for file in empty_images if os.path.isfile(file)]
        full_images = [file for file in tile_images if rename_img2lbl(file) in tile_labels]
        unpaired_labels = [file for file in tile_labels if rename_lbl2img(file) not in tile_images]

        self.log.info(f"{self._class_name}.{'_get_empty_images'}: found {len(empty_images)} empty images and {len(full_images)} full images. ")
        
        if len(unpaired_labels) > 2:
            self.log.warning(f"{class_name}.{func_name}:❗️ Found {len(unpaired_labels)} labels that don't have a matching image. (Ex:{unpaired_labels[0]}). Maybe deleting based on size also deleted images with objects?")

        assert full_images is not None, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'full_images' is None. No full image found. ")

        return (full_images, empty_images)
    

    @abstractmethod
    def __call__(self) -> None:
        return




    def _get_instances_df(self): 
        """ Creates the tiles DataFrame. """
        
        # @log_start_finish(class_name=self.__class__.__name__, func_name='_get_instances_df', msg = f"Creating instances dataframe" )
        def do():
            df = pd.DataFrame(columns=['class_n','width','height','area', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

            # open all labels: 
            label_f = self.data['tile_labels']
            i = 0
            for file in tqdm(label_f, desc = 'Creating df_instances'):

                fn = os.path.basename(file)
                with open(file, 'r') as f:
                    rows = f.readlines()
                # self.log.info(f'file opened')
                
                assert len(rows) <= 30, f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? "
                # self.log.info(f'assertion passed ')
                class_n = [row[0] for row in rows]
                rows = [row.replace('\n', '') for row in rows]
                for row in rows:
                    items = row.split(' ')
                    class_n = items[0]
                    width = float(items[3])
                    height = float(items[-1])
                    instance_n = f"obj_{i}"
                    tile_n = fn.split('sample')[1][1:].split('.')[0]
                    sample_n = fn.split('sample')[1][0]
                    wsi_n = fn.split('_sample')[0]
                    fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]

                    info_dict = {'class_n':class_n, 'width':round((width), 4), 'height':round(height,4), 
                                'area':round(width*height, 4), 'obj_n':instance_n, 'fold':fold, 'tile':tile_n, 
                                'sample':sample_n,'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
                    # self.log.info(f'info dict: {info_dict} ')
                    
                    df.loc[i] = pd.Series(info_dict)
                    i += 1
                self.df_instances = df

            if self.verbose is True:
                # self.log.info(df.loc[:1])
                df.describe()

            self._add_empty2df()

            return df
        
        returned = do()

        return  returned
    

    def _add_empty2df(self) -> None: 
        """ Helper function for _get_tiles_df. Adds to the dataframe also empty tiles."""

        # @log_start_finish(class_name=self.__class__.__name__, func_name='_add_empty2df', msg = f"Adding empty images:" )
        def do():
            _, empty = self._get_empty_images()
            self.log.info(f"empty: {len(empty)}")

            # add empty tiles as new rows to self.df_instances:
            i = len(self.df_instances)
            for file in tqdm(empty, desc = 'Scanning empty tiles'):

                fn = os.path.basename(file)
                tile_n = fn.split('sample')[1][1:].split('.')[0]
                sample_n = fn.split('sample')[1][0]
                wsi_n = fn.split('_sample')[0]
                fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
                info_dict = {'class_n':np.nan, 'width':np.nan, 'height':np.nan, 
                            'area':np.nan, 'obj_n':np.nan, 'fold':fold, 'tile':tile_n, 
                            'sample':sample_n,'wsi':{wsi_n}} # all empty values set to nan
                self.df_instances.loc[i] = pd.Series(info_dict)

                i+=1
            return 
        
        do()

        return  



    def _get_tiles_df(self): 
        func_n = self._get_tiles_df.__name__

        self.format_msg(msg=f"Creating tile dataframe:", func_n=func_n)

        full, empty = self._get_empty_images()

        df = pd.DataFrame(columns=['class_n', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

        # open all labels: 
        label_f = self.data['tile_labels']
        i = 0
        for file in tqdm(label_f, desc = 'Creating tile_df'):
            class_n = 'uhealthy_gloms'

            fn = os.path.basename(file)
            
            # self.log.info(f" label opening")
            with open(file, 'r') as f:
                rows = f.readlines()
            # self.log.info(f" label opened.")

            assert len(rows) <= 30, f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? "
            rows = [row.replace('\n', '') for row in rows]

            # self.log.info(f"rows:{len(rows)}")
            for row in rows:
                # self.log.info(row)
                items = row.split(' ')
                # self.log.info(items)
                class_n = 'healthy' if int(float(items[0])) == 1 else class_n
            # self.log.info(f"rows done")
            n_objs = len(rows)
            tile_n = fn.split('sample')[1][1:].split('.')[0]
            sample_n = fn.split('sample')[1][0]
            wsi_n = fn.split('_sample')[0]
            fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
            # self.log.info(f"paths done")

            info_dict = {'class_n':class_n, 'obj_n':n_objs, 'fold':fold, 'tile':tile_n, 
                        'sample':sample_n,'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
            # self.log.info(f"info_dict:{info_dict}")
            df.loc[i] = pd.Series(info_dict)
            i += 1
        
        # TODO FIX: NOT MACHING BETWEEN DF AND EMPTY/FULL
        # assert len(df) == len(full), f"len(df):{len(df)}, len(full):{len(full)}"
        # check at least one not empty 
        self.log.info(f"adding empty files:")
        # now add also all empty tiles: 
        for file in empty:
            class_n = 'empty'
            n_objs = 0
            tile_n = fn.split('sample')[1][1:].split('.')[0]
            sample_n = fn.split('sample')[1][0]
            wsi_n = fn.split('_sample')[0]
            fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
            info_dict = {'class_n':class_n, 'obj_n':n_objs, 'fold':fold, 'tile':tile_n, 
                        'sample':sample_n,'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
            df.loc[i] = pd.Series(info_dict)
            i += 1

                
        if self.verbose is True:
            self.log.info(df.describe()) 

        
        return df
        

