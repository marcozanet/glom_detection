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
from processor_tile import TileProcessor




class WSI_Processor(TileProcessor):

    def __init__(self,
                src_root: str, 
                dst_root: str, 
                step: int,
                convert_from: Literal['json_wsi_mask', 'jsonliketxt_wsi_mask', 'gson_wsi_mask'], 
                convert_to: Literal['json_wsi_bboxes', 'txt_wsi_bboxes'],
                slide_format: Literal['tiff', 'tif'],
                # label_format: Literal['json', 'gson'],
                ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                safe_copy: bool = True,
                tile_shape: tuple = (4096, 4096),
                verbose: bool = False,
                empty_perc: float = 0.1, 
                ) -> None:

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
        assert task in ['segmentation', 'detection', 'both'], ValueError(f"'task'= {task} should be either segmentation, detection or both. ")
        assert isinstance(tile_shape, tuple), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."
        assert isinstance(step, int), f"'step' should be int."
        assert isinstance(safe_copy, bool), f"safe_copy should be boolean."
        assert isinstance(empty_perc, float), f"'empty_perc' should be a float between 0 and 1."
        assert 0<=empty_perc<=1, f"'empty_perc' should be a float between 0 and 1."
        assert slide_format in ['tiff', 'tif'], f"'slide_format'={slide_format} should be either 'tiff' or 'tif' format."
        # assert label_format in ['json', 'gson'], f"'label_format'={label_format} should be either 'json' or 'gson' format."

        self.src_root = src_root
        self.dst_root = dst_root
        self.ratio = ratio
        self.task = task 
        self.tile_shape = tile_shape
        self.verbose = verbose
        self.step = step
        self.safe_copy = safe_copy
        self.empty_perc = empty_perc
        self.slide_format = slide_format
        self.convert_from = convert_from
        self.convert_to = convert_to

    
    def get_yolo_labels(self) -> None:
        """ Converts .json WSI annotation file to .txt tile annotations suitable to be trained with YOLO.  """
        
        self.log.info(f"Getting YOLO label tiles: ⏳")

        # 1) Conversion
        folder = self.src_root
        converter = Converter(folder = folder, 
                              convert_from=self.convert_from, 
                              convert_to=self.convert_to )
        converter()

        # 2) Label Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= self.tile_shape, 
                      save_root=self.src_root, 
                      step = self.step,
                      verbose = self.verbose)
        tiler(target_format='txt')

        self.log.info(f"Tiled annotations into .txt tiled files: ✅")
        labels_dir = os.path.join(self.src_root, 'labels')

        return labels_dir
    

    def get_yolo_images(self) -> None:
        """ Tiles the WSI and saves the patches in 'save_folder'. """

        self.log.info(f"Getting YOLO image tiles: ⏳")

        # Image Tiling
        tiler = Tiler(folder = self.src_root, 
                      tile_shape= self.tile_shape, 
                      save_root=self.src_root, 
                      step = self.step,
                      verbose = self.verbose)
        tiler(target_format=self.slide_format)

        self.log.info(f"Tiled slide into image tiles: ✅")
        images_dir = os.path.join(self.src_root, 'images')
        
        return images_dir
    

    def _get_yolo_data(self) -> None:
        """ Tiles both WSI and its annotations and saves them into 'images', 'labels' folder.  """
        
        print(f"Tip: Make sure 'step' divides in tiles such that gloms are at least once fully captured in one tile. ")

        images_dir = self.get_yolo_images()
        labels_dir = self.get_yolo_labels()

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

        if self.task == 'segmentation':
            print(f"Splitting for segmentation has not yet been implemented. ")
            raise NotImplementedError()

        elif self.task == 'detection':

            assert empty_images is not None, ValueError(f"'empty_images' is None but should be provided.")
            train_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in train_imgs]
            val_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in val_imgs]
            test_labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in test_imgs]

            # now also add emtpy images to train
            n_tot_imgs =  int(len(train_imgs) * (1+ self.empty_perc))
            n_empty_imgs = int(self.empty_perc * n_tot_imgs)
            additional_empty_imgs = empty_images[:n_empty_imgs]
            for image in additional_empty_imgs:
                train_imgs.append(image)

            labels = [train_labels, val_labels, test_labels]

            return images, labels


    def get_trainvaltest(self) :

        traindir = os.path.join(self.dst_root, self.task, 'train')
        valdir = os.path.join(self.dst_root, self.task, 'val')
        testdir =  os.path.join(self.dst_root, self.task, 'test')

        # 1) check if dataset already exists:
        skip_splitting = self._check_already_splitted()
        if skip_splitting:
            print('Files are already splitted into train, val and test.')
            self.log.info(f"Getting YOLO image tiles: ✅")
            self.log.info(f"Getting YOLO label tiles: ✅")
            self.log.info("Splitting in train, val, test sets ✅. ")
            return traindir, valdir, testdir
        else:
            print(f"clearing dataset")
            self._clear_dataset()

        if self.task == 'segmentation':
            
            print(f"splitting for segmentation task has not yet been implemented.")
            raise NotImplementedError()
            image_list, mask_list = self._get_images_masks()
            images, masks = self._split_images_trainvaltest(image_list = image_list, mask_list= mask_list)
            self._makedirs_moveimages(images=images, masks=masks)
            
        elif self.task == 'detection':

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

        if self.task == 'detection':
            traindir, valdir, testdir = self.get_trainvaltest()
        
        else:
            raise NotImplementedError()

        return traindir, valdir, testdir


def test_WSI_Processor():
    
    print(" ########################    TEST 1: ⏳     ########################")
    # setting:
    src_root = '/Users/marco/Downloads/test_folders/test_processor'
    dst_root = '/Users/marco/Downloads/test_folders/test_processor'
    ratio = [0.7, 0.15, 0.15]
    task = 'detection'
    convert_from = 'gson_wsi_mask'
    convert_to = 'txt_wsi_bboxes'    
    empty_perc =  0.1 
    step = 1024
    slide_format= 'tif'
    tile_shape = (4096, 4096)
    verbose = False
    # make dirs:
    os.makedirs(src_root, exist_ok=True)
    os.makedirs(dst_root, exist_ok=True)
    # run testing
    processor = WSI_Processor(src_root = src_root, 
                              dst_root = dst_root,
                              convert_from = convert_from, 
                              convert_to = convert_to,                               
                              ratio = ratio, 
                              slide_format= slide_format,
                              task = task, 
                              step = step,
                              tile_shape = tile_shape,
                              empty_perc = empty_perc,
                              verbose = verbose)
    processor()
    print(" ########################    TEST 2: ✅    ########################")

    return  



if __name__ == '__main__':
    test_WSI_Processor()