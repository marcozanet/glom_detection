import os 
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np
import pandas as pd 
from tqdm import tqdm


class Profiler(): 

    def __init__(self, 
                data_root:str, 
                wsi_images_like:str = '*.tif', 
                wsi_labels_like:str = '*_sample?.txt',
                tile_images_like:str = '*sample*.png',
                tile_labels_like:str = '*sample*.txt') -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            wsi_image_like= e.g. '*.tif'
            wsi_label_like = e.g. '*_sample?.txt'
            tile_image_like = e.g. '*sample*.png'
            tile_label_like = e.g. '*sample*.txt'
            '"""

        assert os.path.isdir(data_root), f"'data_root':{data_root} is not a valid dirpath."
        self.data_root = data_root
        self.wsi_images_like = wsi_images_like
        self.wsi_labels_like = wsi_labels_like
        self.tile_images_like = tile_images_like
        self.tile_labels_like = tile_labels_like
        self.wsi_image_format = wsi_images_like.split('.')[-1]
        self.wsi_label_format = wsi_labels_like.split('.')[-1]
        self.tiles_image_format = tile_images_like.split('.')[-1]
        self.tiles_label_format = tile_labels_like.split('.')[-1]
        self.data = self._get_data()

        return


    
    def _get_data(self) -> dict:
        """ From a roots like root -> wsi/tiles->train,val,test->images/labels, 
            it returns a list of wsi images/labels and tiles images/labels."""

        wsi_images = glob(os.path.join(self.data_root, 'wsi', '*', 'images', self.wsi_images_like))
        wsi_labels = glob(os.path.join(self.data_root, 'wsi', '*', 'labels', self.wsi_labels_like))
        tile_images = glob(os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like))
        tile_labels = glob(os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like))
        data = {'wsi_images':wsi_images, 'wsi_labels':wsi_labels, 'tile_images':tile_images,  'tile_labels':tile_labels }

        return data


    
    def _get_unique_labels(self, verbose = False) -> dict:
        """ Returns unique label values from the dataset. """

        unique_labels = []
        for label_fp in self.data['tile_labels']:
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                labels = [row[0] for row in rows]
                unique_labels.extend(labels)
        unique_labels = list(set(unique_labels))
        
        if verbose is True:
            print(f"Unique classes: {unique_labels}")

        return unique_labels


    
    def _get_class_freq(self) -> dict:
        
        class_freq = {'0':0, '1':0, '2':0, '3':0}
        for label_fp in self.data['tile_labels']:
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                labels = [row[0] for row in rows]
                for label in labels:
                    class_freq[label] += 1
        print(f"class_freq: {class_freq}")

        return class_freq
    
    
    def _get_df(self): 

        df = pd.DataFrame(columns=['class_n','width','height','area', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

        # open all labels: 
        label_f = self.data['tile_labels']
        i = 0
        for file in tqdm(label_f, desc = 'Scanning data'):

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
        print(df.head())
        # self._add_empty2df()

        return  df
    

    def _add_empty2df(self): 

        _, empty = self._get_empty_images()

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


    def _get_tile_df(self): 

        full, empty = self._get_empty_images()

        df = pd.DataFrame(columns=['class_n','width','height','area', 'obj_n', 'tile', 'fold', 'sample', 'wsi', 'fn' ])

        # open all labels: 
        label_f = self.data['tile_labels']
        i = 0
        for file in tqdm(label_f, desc = 'Scanning data'):

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
        print(df.head())

        
        return  

    def _get_nsamples_inslide(self, wsi_fp:str, verbose:bool = False) -> tuple: 
        """ Computes number of samples of a given slide. """
        assert os.path.isfile(wsi_fp), f"'wsi_fp':{wsi_fp} is not a valid filepath."

        wsi_fn = os.path.basename(wsi_fp).replace(f'.{self.wsi_image_format}', '')
        labels = [label for label in self.data['wsi_labels'] if wsi_fn in label]
        n_samples = len(labels)

        if verbose is True:
            print(f"wsi_fn: '{wsi_fn}.{self.wsi_image_format}' has {n_samples} samples.")

        return (wsi_fn, n_samples)


    def _get_ngloms_inslide(self, wsi_fp:str, verbose:bool = False) -> list: 
        """ Computes number of samples of a given slide. """
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
                print(f"sample_{sample_i} in wsi_fn: '{wsi_fn}.{self.wsi_image_format}' has {n_gloms} gloms.")
            
            samples_gloms.append((sample_fn, n_gloms))
        
        return samples_gloms     

    
    def _get_gloms_samples(self):

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


    def _get_empty_images(self):

        tile_images = self.data['tile_images']
        tile_labels = self.data['tile_labels']

        empty_images = [file for file in tile_images if os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format,self.tiles_label_format)) not in tile_labels]
        empty_images = [file for file in empty_images if "DS" not in empty_images and self.tiles_image_format in file]
        empty_images = [file for file in empty_images if os.path.isfile(file)]

        full_images = [file for file in tile_images if os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format,self.tiles_label_format)) in tile_labels]
        unpaired_labels = [file for file in tile_labels if os.path.join(os.path.dirname(file).replace('labels', 'images'), os.path.basename(file).replace(self.tiles_label_format, self.tiles_image_format)) not in tile_images]
        
        if len(unpaired_labels) > 2:
            print(f"❗️ Found {len(unpaired_labels)} labels that don't have a matching image. Maybe deleting based on size also deleted images with objects?")

        return full_images, empty_images
    
    
    def show_summary(self): 


        return

    
    def __call__(self) -> None:

        self.df = self._get_df()
        self._get_tile_df()


        return



def test_Profiler():
    system = 'mac'
    data_root = '/Users/marco/Downloads/train_20feb23' if system == 'mac' else r'C:\marco\biopsies\muw\detection'
    profiler = Profiler(data_root=data_root)
    profiler()

    return


if __name__ == '__main__':
    test_Profiler()