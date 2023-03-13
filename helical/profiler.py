import os 
from glob import glob
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd 
from tqdm import tqdm
# from loggers import get_logger
from decorators import log_start_finish
from configurator import Configurator


class Profiler(Configurator): 

    def __init__(self, 
                data_root:str, 
                wsi_images_like:str = '*.tif', 
                wsi_labels_like:str = '*_sample?.txt',
                tile_images_like:str = '*sample*.png',
                tile_labels_like:str = '*sample*.txt',
                empty_ok:bool = False,
                verbose:bool=False) -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            wsi_image_like= e.g. '*.tif'
            wsi_label_like = e.g. '*_sample?.txt'
            tile_image_like = e.g. '*sample*.png'
            tile_label_like = e.g. '*sample*.txt'
            '"""
        # self.log = get_logger()
        super().__init__()
        assert os.path.isdir(data_root), f"'data_root':{data_root} is not a valid dirpath."

        self._class_name = self.__class__.__name__
        self.data_root = data_root
        self.wsi_images_like = wsi_images_like
        self.wsi_labels_like = wsi_labels_like
        self.tile_images_like = tile_images_like
        self.tile_labels_like = tile_labels_like
        self.wsi_image_format = wsi_images_like.split('.')[-1]
        self.wsi_label_format = wsi_labels_like.split('.')[-1]
        self.tiles_image_format = tile_images_like.split('.')[-1]
        self.tiles_label_format = tile_labels_like.split('.')[-1]
        self.verbose = verbose
        self.empty_ok = empty_ok


        self.data = self._get_data()
        self.log.info(f"len data images: {self.data['tile_images']}")


        return


    
    def _get_data(self) -> dict:
        """ From a roots like root -> wsi/tiles->train,val,test->images/labels, 
            it returns a list of wsi images/labels and tiles images/labels."""
        
        # @log_start_finish(class_name=self.__class__.__name__, func_name='_get_data', 
                        #   msg = f" Getting data from: '{os.path.basename(self.data_root)}'" )
        def do():

            wsi_images = glob(os.path.join(self.data_root, 'wsi', '*', 'images', self.wsi_images_like))
            wsi_labels = glob(os.path.join(self.data_root, 'wsi', '*', 'labels', self.wsi_labels_like))
            tile_images = glob(os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like))
            tile_labels = glob(os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like))
            self.log.info(f"looking for images like: {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} ")
            self.log.info(f"images found: {len(tile_images)}")
            data = {'wsi_images':wsi_images, 'wsi_labels':wsi_labels, 'tile_images':tile_images,  'tile_labels':tile_labels }
            
            if self.empty_ok is False:
                assert len(tile_images) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
            else:
                if len(tile_images) > 0: 
                    self.log.warning(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
            if self.empty_ok is False:
                assert len(tile_labels) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
            else:
                if len(tile_labels) > 0:
                    self.log.warning(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
            
            self.log.info(f"completed func, returning {data}")
            return data
        
        returned = do()
        self.log.info(f"do called. returning {returned}")

        return returned


    
    def _get_unique_labels(self, verbose = False) -> dict:
        """ Returns unique label values from the dataset. """
        class_name = self.__class__.__name__
        func_name = '_get_unique_labels'

        # @log_start_finish(class_name, func_name, msg = f"Getting unique labels:'{os.path.basename(self.data_root)} '" )
        def do():
            unique_labels = []
            for label_fp in self.data['tile_labels']:
                with open(label_fp, mode ='r') as f:
                    rows = f.readlines()
                    labels = [row[0] for row in rows]
                    unique_labels.extend(labels)
            unique_labels = list(set(unique_labels))
            
            if verbose is True:
                self.log.info(f"{class_name}.{func_name}: Unique classes: {unique_labels}", )

            return unique_labels
        
        returned = do()

        return  returned


    
    def _get_class_freq(self) -> dict:

        class_name = self.__class__.__name__
        func_name = '_get_class_freq'

        @log_start_finish(class_name=class_name, func_name=func_name,  msg = f" Getting classes frequency" )
        def do():
            class_freq = {'0':0, '1':0, '2':0, '3':0}
            for label_fp in self.data['tile_labels']:
                with open(label_fp, mode ='r') as f:
                    rows = f.readlines()
                    labels = [row[0] for row in rows]
                    for label in labels:
                        class_freq[label] += 1

            if self.verbose is True:
                self.log.info(f"{class_name}.{func_name}: class_freq: {class_freq}")

            return class_freq
        returned = do()

        return returned
    
    
    def _get_instances_df(self): 
        """ Creates the tiles DataFrame. """
        
        @log_start_finish(class_name=self.__class__.__name__, func_name='_get_tiles_df', msg = f"Creating instances dataframe" )
        def do():
            df = pd.DataFrame(columns=['class_n','width','height','area', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

            # open all labels: 
            label_f = self.data['tile_labels']
            i = 0
            for file in tqdm(label_f, desc = 'Creating df_instances'):

                fn = os.path.basename(file)
                
                with open(file, 'r') as f:
                    rows = f.readlines()
                
                assert len(rows) <= 30, f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? "
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
                    df.loc[i] = pd.Series(info_dict)
                    i += 1
                self.df = df

            if self.verbose is True:
                # self.log.info(df.loc[:1])
                df.describe()

            self._add_empty2df()

            return df
        
        returned = do()

        return  returned
    

    def _add_empty2df(self) -> None: 
        """ Helper function for _get_tiles_df. Adds to the dataframe also empty tiles."""

        @log_start_finish(class_name=self.__class__.__name__, func_name='_add_empty2df', msg = f"Adding empty images:" )
        def do():
            _, empty = self._get_empty_images()
            self.log.info(f"empty: {len(empty)}")

            # add empty tiles as new rows to self.df:
            i = len(self.df)
            for file in tqdm(empty, desc = 'Scanning empty tiles'):

                fn = os.path.basename(file)
                tile_n = fn.split('sample')[1][1:].split('.')[0]
                sample_n = fn.split('sample')[1][0]
                wsi_n = fn.split('_sample')[0]
                fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
                info_dict = {'class_n':np.nan, 'width':np.nan, 'height':np.nan, 
                            'area':np.nan, 'obj_n':np.nan, 'fold':fold, 'tile':tile_n, 
                            'sample':sample_n,'wsi':{wsi_n}} # all empty values set to nan
                self.df.loc[i] = pd.Series(info_dict)

                i+=1
            return 
        
        do()

        return  



    def _get_tiles_df(self): 
        
        @log_start_finish(class_name=self.__class__.__name__, func_name='_get_tiles_df', msg = f"Creating tile dataframe:" )
        def do():

            full, empty = self._get_empty_images()

            df = pd.DataFrame(columns=['class_n', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

            # open all labels: 
            label_f = self.data['tile_labels']
            i = 0
            for file in tqdm(label_f, desc = 'Creating tile_df'):
                class_n = 'uhealthy_gloms'

                fn = os.path.basename(file)
                
                with open(file, 'r') as f:
                    rows = f.readlines()
                
                assert len(rows) <= 30, f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? "
                rows = [row.replace('\n', '') for row in rows]

                for row in rows:
                    items = row.split(' ')
                    class_n = 'healthy' if int(items[0]) == 1 else class_n
                n_objs = len(rows)
                tile_n = fn.split('sample')[1][1:].split('.')[0]
                sample_n = fn.split('sample')[1][0]
                wsi_n = fn.split('_sample')[0]
                fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
                info_dict = {'class_n':class_n, 'obj_n':n_objs, 'fold':fold, 'tile':tile_n, 
                            'sample':sample_n,'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
                df.loc[i] = pd.Series(info_dict)
                i += 1
            
            # TODO FIX: NOT MACHING BETWEEN DF AND EMPTY/FULL
            # assert len(df) == len(full), f"len(df):{len(df)}, len(full):{len(full)}"
            # check at least one not empty 

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
        
        returned = do()

        return returned

    def _get_nsamples_inslide(self, wsi_fp:str, verbose:bool = False) -> tuple: 
        """ Computes number of samples of a given slide. """
        class_name = self.__class__.__name__
        func_name = '_get_nsamples_inslide'
       
        @log_start_finish(class_name=class_name, func_name=func_name, msg = f"Getting nsamples in slide:" )
        def do():
            assert os.path.isfile(wsi_fp), f"'wsi_fp':{wsi_fp} is not a valid filepath."

            wsi_fn = os.path.basename(wsi_fp).replace(f'.{self.wsi_image_format}', '')
            labels = [label for label in self.data['wsi_labels'] if wsi_fn in label]
            n_samples = len(labels)

            if verbose is True:
                self.log.info(f"{class_name}.{func_name}: wsi_fn: '{wsi_fn}.{self.wsi_image_format}' has {n_samples} samples.")

            return (wsi_fn, n_samples)
        
        returned = do()

        return returned


    def _get_ngloms_inslide(self, wsi_fp:str, verbose:bool = False) -> list: 
        """ Computes number of glomeruli within a given slide. """
        class_name = self.__class__.__name__
        func_name = '_get_ngloms_inslide'        
        
        @log_start_finish(class_name=class_name, func_name=func_name, msg = f"Getting ngloms in slide:" )
        def do():
            assert os.path.isfile(wsi_fp), f"'wsi_fp':{wsi_fp} is not a valid filepath."

            wsi_fn = os.path.basename(wsi_fp).replace(f'.{self.wsi_image_format}', '')
            labels = [label for label in self.data['wsi_labels'] if wsi_fn in label]

            samples_gloms = []
            for sample_i in labels: 

                with open(sample_i, 'r') as f:
                    rows = f.readlines()
                n_gloms = len(rows)
                sample_fn = os.path.basename(sample_i).split('.')[0]

                if verbose is True:
                    self.log.info(f"{class_name}.{func_name}: sample_{sample_i} in wsi_fn: '{wsi_fn}.{self.wsi_image_format}' has {n_gloms} gloms.")
                
                samples_gloms.append((sample_fn, n_gloms))
            
            return samples_gloms     

        returned = do()

        return returned

    
    def _get_gloms_samples(self):

        @log_start_finish(class_name=self.__class__.__name__, func_name='_get_gloms_samples', msg = f"Getting nsamples in slide:" )
        def do():
            old_list = list(map(self._get_ngloms_inslide, self.data['wsi_images']))
            # unpacking all tuples
            new_list = [] 
            for el in old_list: 
                if len(el) > 2:
                    for subel in el: 
                        new_list.extend([subel])
                else:
                    new_list.extend(el)

            return new_list
        returned = do()

        return returned


    def _get_empty_images(self):
        """ Return """

        class_name = self.__class__.__name__
        func_name = '_get_empty_images'

        @log_start_finish(class_name=self.__class__.__name__, func_name=func_name, msg = f"Getting nsamples in slide:" )
        def do():
            tile_images = self.data['tile_images']
            tile_labels = self.data['tile_labels']

            assert len(tile_images)>0, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'tile_images':{len(tile_images)}. No tile image found. ")
            assert len(tile_labels)>0, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'tile_labels':{len(tile_labels)}. No tile label found. ")
            
            empty_images = [file for file in tile_images if os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format,self.tiles_label_format)) not in tile_labels]
            empty_images = [file for file in empty_images if "DS" not in empty_images and self.tiles_image_format in file]
            empty_images = [file for file in empty_images if os.path.isfile(file)]

            full_images = [file for file in tile_images if os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format,self.tiles_label_format)) in tile_labels]
            unpaired_labels = [file for file in tile_labels if os.path.join(os.path.dirname(file).replace('labels', 'images'), os.path.basename(file).replace(self.tiles_label_format, self.tiles_image_format)) not in tile_images]
            
            if len(unpaired_labels) > 2:
                self.log.info(f"{class_name}.{func_name}:❗️ Found {len(unpaired_labels)} labels that don't have a matching image. Maybe deleting based on size also deleted images with objects?")

            assert full_images is not None, self.log.error(f"{self._class_name}.{'_get_empty_images'}: 'full_images' is None. No full image found. ")

            return (full_images, empty_images)
        
        returned = do()

        return returned
    

    def log_data_summary(self): 
        """ Logs a summary of the data. """

        self.log.info(f"{self._class_name}.{'_log_summary'}: 'Data root':{self.data_root}")
        self.data=self._get_data()
        slides = self.data['wsi_images']
        train_slides = [file for file in slides if 'train' in file]
        val_slides = [file for file in slides if 'val' in file]
        test_slides = [file for file in slides if 'test' in file]
        self.log.info(f"{self._class_name}.{'_log_summary'}: 'trainset':{train_slides}, 'valset':{val_slides}, 'testset':{test_slides}")
        class_freq = self._get_class_freq()
        self.log.info(f"{self._class_name}.{'_log_summary'}: 'class frequency':{class_freq} ")
        unique_labels = self._get_unique_labels()
        self.log.info(f"{self._class_name}.{'_log_summary'}: 'unique labels':{unique_labels} ")
        returned = self._get_empty_images()
        full_images, empty_images = returned[0], returned[1]
        empty_perc = round(len(empty_images)/(len(full_images) + len(empty_images)), 2)
        self.log.info(f"{self._class_name}.{'_log_summary'}: 'full_images':{len(full_images)}. 'empty_images':{len(empty_images)}. Empty perc:{empty_perc} ")

        return


    
    
    def show_summary(self): 


        return

    
    def __call__(self) -> None:

        self.df = self._get_instances_df()
        self._get_tiles_df()
        self._get_class_freq()
        self._get_empty_images()


        return



def test_Profiler():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    verbose = True
    data_root = '/Users/marco/Downloads/train_20feb23_copy' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    profiler = Profiler(data_root=data_root, verbose=True)
    profiler()

    return


if __name__ == '__main__':
    test_Profiler()