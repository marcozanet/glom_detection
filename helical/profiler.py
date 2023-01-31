import os 
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np
import pandas as pd 


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
        self.samples_slides = list(map(self._get_nsamples_inslide, self.data['wsi_images']))
        self.gloms_slides = self._get_gloms_samples()
        # print(self.gloms_slides)
        self.show_summary()

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
    
    def show_summary(self): 
        
        # 1) Gloms per tissue sample:
        df = pd.DataFrame(data = self.gloms_slides, columns = ['sample', 'n_gloms'])
        fig = sns.barplot(df, x = df.index, y = 'n_gloms')
        # fig = sns.displot(n_gloms, bins = range(np.array(n_gloms).max()), kde = True)
        plt.title('Barplot #gloms per tissue sample')
        plt.xlabel('#gloms_per_sample')


        return


    # def get_glom_per_sample(self):

    #     # 1) get correspondent label
    #     for wsi_fn, n_sample in 



    #     return
 



def test_Profiler():
    system = 'mac'
    data_root = '/Users/marco/Downloads/try_train/detection' if system == 'mac' else r'C:\marco\biopsies\muw\detection'
    profiler = Profiler(data_root=data_root)

    return


if __name__ == '__main__':
    test_Profiler()