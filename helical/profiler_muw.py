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
from profiler_base import ProfilerBase


class ProfilerMUW(ProfilerBase): 

    def __init__(self, 
                *args,
                **kwargs) -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            wsi_image_like= e.g. '*.tif'
            wsi_label_like = e.g. '*_sample?.txt'
            tile_image_like = e.g. '*sample*.png'
            tile_label_like = e.g. '*sample*.txt'
            '"""
        

        other_params = {'wsi_images_like':'*.tif', 'wsi_labels_like': '*_sample?.txt', 
                        'tile_images_like':'*sample*.png', 'tile_labels_like':'*sample*.txt'}
        kwargs.update(other_params)
        super().__init__(*args, **kwargs)

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




    
    def __call__(self) -> None:

        self.df_instances = self._get_instances_df()
        self._get_tiles_df()
        self._get_class_freq()
        self._get_empty_images()


        return



