import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from typing import List, Tuple, Literal
import numpy as np
import random
from conversion import Converter
from tiling import Tiler
from loggers import get_logger
from processor_tile import Tile_Processor




class WSI_Processor(Tile_Processor):

    def __init__(self,
                src_root: str, 
                dst_root: str, 
                ratio = [0.7, 0.15, 0.15], 
                mode = Literal['detection', 'segmentation', 'both'],
                tile_shape: tuple = (4096, 4096),
                empty_perc: float = 0.1) -> None:

        """ Sets paths and folders starting from tiles. 
            NB images should be named <name>.png <name-labelled>.png img_folder:"""

        self.log = get_logger()

        assert isinstance(src_root, str), TypeError(f"src_root type is {type(src_root)}, but should be str.")
        assert os.path.isdir(src_root), ValueError(f"{src_root} is not a dir. ")
        assert isinstance(dst_root, str), TypeError(f"dst_root type is {type(dst_root)}, but should be str.")
        assert os.path.isdir(dst_root), ValueError(f"{dst_root} is not a dir. ")
        assert isinstance(ratio, List), TypeError(f"'ratio' should be left empty or be a list. ")
        try:    
            ratio = [float(value) if isinstance(value, str) else value for value in ratio]
        except:
            TypeError(f"Values in 'ratio' can't be converted to float.")
        assert len(ratio) == 3 and round(np.sum(np.array(ratio)), 2) == 1.0, ValueError(f"'ratio' should be a list of floats with sum 1, but has sum {np.sum(np.array(ratio))}." )
        assert mode in ['segmentation', 'detection', 'both'], ValueError(f"'mode'= {mode} should be either segmentation, detection or both. ")
        assert isinstance(tile_shape, tuple), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape' should be a tuple of int.")
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.ratio = ratio
        self.mode = mode 
        self.empty_perc = 0.1 if empty_perc is None else empty_perc
        self.tile_shape = tile_shape

    
    def get_yolo_labels(self, tile_shape: tuple = (4096, 4096)) -> None:
        """ Converts .json WSI annotation file to .txt tile annotations suitable to be trained with YOLO.  """
        
        self.log.info(f"Getting YOLO label tiles: ⏳")

        # 1) Conversion
        folder = self.src_root
        converter = Converter(folder = folder, 
                              convert_from='json_wsi_mask', 
                              convert_to='txt_wsi_bboxes' )
        converter()

        # 2) Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= tile_shape, 
                      save_root=self.src_root)
        tiler(target_format='txt')

        self.log.info(f"Getting YOLO image tiles: ✅")
        labels_dir = os.path.join(self.src_root, 'labels')

        return labels_dir
    

    def get_yolo_images(self, tile_shape: tuple = (4096, 4096) ) -> None:
        """ Tiles the WSI and saves the patches in 'save_folder'. """

        self.log.info(f"Getting YOLO image tiles: ⏳")

        # 1) Conversion
        folder = self.src_root
        converter = Converter(folder = folder, 
                              convert_from='json_wsi_mask', 
                              convert_to='txt_wsi_bboxes' )
        converter()

        # 2) Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= tile_shape, 
                      save_root=self.src_root)
        tiler(target_format='tiff')

        self.log.info(f"Getting YOLO image tiles: ✅")
        images_dir = os.path.join(self.src_root, 'images')
        
        return images_dir
    

    def _get_yolo_data(self) -> None:
        """ Tiles both WSI and its annotations and saves them into 'images', 'labels' folder.  """

        images_dir = self.get_yolo_images(tile_shape=self.tile_shape)
        labels_dir = self.get_yolo_labels(tile_shape=self.tile_shape)

        return images_dir, labels_dir
    

    def _split_images_trainvaltest(self, 
                                   image_list: list, 
                                   mask_list: list = None, 
                                   empty_images: list = None,
                                   train_balance: bool = False) -> Tuple[List, List] :
        """ Splits images in train, val and test. 
            Returns: tuple containing lists of train, val, test images and masks"""

        n_images = len(image_list)
        n_val_imgs = int(self.ratio[1] * n_images)
        n_test_imgs = int(self.ratio[2] * n_images)

        # 1) randomly pick val images:
        val_imgs = []
        for _ in range(n_val_imgs):
            rand_img = random.choice(image_list)
            val_imgs.append(rand_img)
            image_list.remove(rand_img)
        print(f"Val images: {len(val_imgs)}")
        
        # 2) randomly pick test images:
        test_imgs = []
        for _ in range(n_test_imgs):
            rand_img = random.choice(image_list)
            test_imgs.append(rand_img)
            image_list.remove(rand_img)
        print(f"Test images: {len(test_imgs)}")

        # 3) remaining images are train:
        if train_balance is False:
            train_imgs = image_list
            print(f"Train images: {len(train_imgs)}")
        else:
            raise NotImplementedError()


        images = [train_imgs, val_imgs, test_imgs]

        if self.mode == 'segmentation':
            
            raise NotImplementedError()

        elif self.mode == 'detection':

            assert empty_images is not None, ValueError(f"'empty_images' is None but should be provided.")
            train_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in train_imgs]
            val_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in val_imgs]
            test_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in test_imgs]

            # now also add emtpy images to train
            n_tot_imgs =  len(train_imgs) / self.empty_perc
            n_empty_imgs = int(self.empty_perc * n_tot_imgs)
            additional_empty_imgs = empty_images[:n_empty_imgs]
            for image in additional_empty_imgs:
                train_imgs.append(image)

            labels = [train_labels, val_labels, test_labels]

            return images, labels


    def get_trainvaltest(self) :

        traindir = os.path.join(self.dst_root, self.mode, 'train')
        valdir = os.path.join(self.dst_root, self.mode, 'val')
        testdir =  os.path.join(self.dst_root, self.mode, 'test')

        # 1) check if dataset already exists:
        skip_splitting = self._check_already_splitted()
        if skip_splitting:
            print('Files are already splitted into train, val and test.')
            self.log.info(f"Getting YOLO image tiles: ✅")
            self.log.info(f"Getting YOLO label tiles: ✅")
            self.log.info("Splitting in train, val, test sets ✅. ")
            return traindir, valdir, testdir
        else:
            self._clear_dataset()

        if self.mode == 'segmentation':
            raise NotImplementedError()
            image_list, mask_list = self._get_images_masks()
            images, masks = self._split_images_trainvaltest(image_list = image_list, mask_list= mask_list)
            self._makedirs_moveimages(images=images, masks=masks)
            
        elif self.mode == 'detection':

            images_dir, labels_dir = self._get_yolo_data() 
            images_list = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if 'DS' not in file]
            full_fnames = [file.split('.')[0] for file in os.listdir(labels_dir)] # images with labelled obj
            full_images = [os.path.join(images_dir, f"{fname}.png") for fname in full_fnames]
            empty_images = [file for file in images_list if file not in full_images]
            print(f"full images: {len(full_images)}, empty images: {len(empty_images)}")


            images, labels = self._split_images_trainvaltest(image_list = full_images, empty_images= empty_images)
            self._makedirs_moveimages(images=images, labels=labels)


        assert os.path.isdir(traindir), f"{traindir} is not a dir."
        assert os.path.isdir(valdir), f"{valdir} is not a dir."
        assert os.path.isdir(testdir), f"{testdir} is not a dir."

        self.log.info("Splitting in train, val, test sets ✅. ")

        return traindir, valdir, testdir


    def __call__(self) :

        if self.mode == 'detection':
            self.get_trainvaltest()
        
        else:
            raise NotImplementedError()



        return


def test_WSI_Processor():
    
    src_root = '/Users/marco/Downloads/new_source'
    dst_root = '/Users/marco/Downloads/folder_random'
    ratio = [0.7, 0.15, 0.15]
    mode = 'detection'
    empty_perc =  0.3

    processor = WSI_Processor(src_root=src_root, 
                              dst_root=dst_root, 
                              ratio=ratio, 
                              mode = mode, 
                              empty_perc=empty_perc)

    processor()

    return



if __name__ == '__main__':
    test_WSI_Processor()
