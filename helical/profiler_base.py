import os 
from glob import glob
from typing import List
from configurator import Configurator
from abc import ABC, abstractmethod


class ProfilerBase(Configurator, ABC): 

    def __init__(self, 
                data_root:str, 
                wsi_images_like:str,
                wsi_labels_like:str,
                tile_images_like:str,
                tile_labels_like:str,
                empty_ok:bool = False,
                verbose:bool=False) -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            wsi_image_like= e.g. '*.tif'
            wsi_label_like = e.g. '*_sample?.txt'
            tile_image_like = e.g. '*sample*.png'
            tile_label_like = e.g. '*sample*.txt'
            '"""
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


        return

    
    def _get_data(self) -> dict:
        """ From a roots like root -> wsi/tiles->train,val,test->images/labels, 
            it returns a list of wsi images/labels and tiles images/labels."""
        
        # look for files;
        wsi_images = glob(os.path.join(self.data_root, 'wsi', '*', 'images', self.wsi_images_like))
        wsi_labels = glob(os.path.join(self.data_root, 'wsi', '*', 'labels', self.wsi_labels_like))
        tile_images = glob(os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like))
        tile_labels = glob(os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like))

        data = {'wsi_images':wsi_images, 'wsi_labels':wsi_labels, 'tile_images':tile_images,  'tile_labels':tile_labels }
        
        # if allowed to be empty:
        if self.empty_ok is False:
            assert len(tile_images) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
            assert len(tile_labels) > 0, self.log.error(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
        else:
            if len(tile_images) > 0: 
                self.log.warning(f"{self._class_name}.{'_get_data'}: no tile image like {os.path.join(self.data_root, 'tiles', '*', 'images', self.tile_images_like)} was found.")
            if len(tile_labels) > 0:
                self.log.warning(f"{self._class_name}.{'_get_data'}: no tile label like {os.path.join(self.data_root, 'tiles', '*', 'labels', self.tile_labels_like)} was found.")
        self.log.info(f"{self._class_name}.{'_get_data'}: Found {len(wsi_images)} slides with {len(wsi_labels)} annotations and {len(wsi_images)} tiles with {len(wsi_labels)} annotations. ")
        
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
    

    def _get_empty_images(self):
        """ Returns full and empty images from the dataset. """

        class_name = self.__class__.__name__
        func_name = '_get_empty_images'

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
    def _get_instances_df(self): 
        """ Creates the tiles DataFrame. """
        return
        

    @abstractmethod
    def _get_tiles_df(self): 
        return



    @abstractmethod
    def __call__(self) -> None:
        return



def test_Profiler():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    verbose = True
    data_root = '/Users/marco/Downloads/train_20feb23_copy' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    profiler = ProfilerBase(data_root=data_root, verbose=True)
    profiler()

    return


if __name__ == '__main__':
    test_Profiler()