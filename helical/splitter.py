import os 
from typing import Literal, List, Tuple
import numpy as np
from loggers import get_logger
import random
from glob import glob 
import shutil
from tqdm import tqdm


class Splitter():

    def __init__(self,
                 src_dir: str,
                 dst_dir: str,
                 image_format: Literal['tif', 'tiff', 'png'],
                #  label_format: Literal['txt', 'gson', 'json'],
                 empty_perc: float = 0.1,
                 ratio = [0.7, 0.15, 0.15], 
                 task = Literal['detection', 'segmentation', 'both'],
                 safe_copy:bool = True,
                 verbose:bool = False) -> None:

        self.log = get_logger()

        ALLOWED_IMAGE_FORMATS = ['tif', 'tiff', 'png']
        ALLOWED_LABEL_FORMATS = ['txt', 'gson', 'json']

        assert image_format in ALLOWED_IMAGE_FORMATS, f"'image_format':{image_format} should be one of {ALLOWED_IMAGE_FORMATS}"
        # assert label_format in ALLOWED_LABEL_FORMATS, f"'label_format':{label_format} should be one of {ALLOWED_LABEL_FORMATS}"
        assert os.path.isdir(dst_dir), f"'dst_dir':{dst_dir} is not a valid filepath."
        assert os.path.isdir(src_dir), f"'src_dir':{src_dir} is not a valid filepath."
        assert task in ['segmentation', 'detection', 'both'], ValueError(f"'task'= {task} should be either segmentation, detection or both. ")
        assert isinstance(ratio, List), TypeError(f"'ratio' should be left empty or be a list. ")
        try:    
            ratio = [float(value) if isinstance(value, str) else value for value in ratio]
        except:
            TypeError(f"Values in 'ratio' can't be converted to float.")
        assert (len(ratio) == 3 or len(ratio) == 2) and round(np.sum(np.array(ratio)), 2) == 1.0, ValueError(f"'ratio' should be a list of floats with sum 1, but has sum {np.sum(np.array(ratio))}." )
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."
        assert isinstance(safe_copy, bool), f"safe_copy should be boolean."

        self.empty_perc = 0.1 if empty_perc is None else empty_perc
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.image_format = image_format
        # self.label_format = label_format
        self.task = task
        self.splitting_mode = 'wtest' if len(ratio)==3 else 'wotest'
        self.image_type = 'wsi' if image_format in ['tif', 'tiff'] else 'tile'
        self.verbose = verbose
        self.ratio = ratio
        self.safe_copy = safe_copy

        
        return
    
    def _split_yolo_tiles(self):
        """ Splits images and annotation tiles into folds."""

        return
    
    def _split_yolo_wsi(self):
        """ Splits WSIs """

        # 1) get slides
        slides = self._get_files(format=self.image_format)
        print(slides)
        # getting all files with those names in src.folder (i.e. all labels, regardless of format)
        labels = [os.path.join(self.src_dir, file) for file in os.listdir(self.src_dir) for slide in slides if os.path.basename(slide.split('.')[0]) in file]
        labels = [file for file in labels if self.image_format not in file]
        print(labels)
        # 2) check if already splitted
        skipping = self._check_already_splitted()
        print(skipping)
        if skipping is True:
            return
        # 3) clear dataset/create_folders:
        dataset_dir = self._make_folders()
        # 4) split 
        images, labels = self._split_slides(image_list = slides, 
                                            label_list = labels)
        # 5) move to new dataset
        self._move_files(images = images, labels = labels, dataset_dir=dataset_dir)

        return
    
    def _split_slides(self, 
                    image_list: list, 
                    label_list: list = None, 
                    train_balance: bool = False) -> Tuple[List, List] :
        """ Splits images in train, val and test. 
            Returns: tuple containing lists of train, val, test images and labels"""

        n_images = len(image_list)
        n_val_imgs = round(self.ratio[1] * n_images) 
        n_test_imgs = int(self.ratio[2] * n_images) if self.splitting_mode == 'wtest' else None

        # 1) randomly pick val images:
        val_imgs = []
        for _ in range(n_val_imgs):
            rand_img = random.choice(image_list)
            val_imgs.append(rand_img)
            image_list.remove(rand_img)
        print(f"Val images: {len(val_imgs)}")
        
        # 2) randomly pick test images:
        if self.splitting_mode == 'wtest':
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


        images = [train_imgs, val_imgs, test_imgs] if self.splitting_mode == 'wtest' else [train_imgs, val_imgs]

        if self.task == 'segmentation':
            print(f"Splitting for segmentation has not yet been implemented. ")
            raise NotImplementedError()

        elif self.task == 'detection':
            train_labels = [file for file in label_list for image in train_imgs if os.path.basename(image).split('.')[0] in file]
            val_labels = [file for file in label_list for image in val_imgs if os.path.basename(image).split('.')[0] in file]
            if self.splitting_mode == 'wtest':
                test_labels = [file for file in label_list for image in test_imgs if os.path.basename(image).split('.')[0] in file]

            labels = [train_labels, val_labels, test_labels] if self.splitting_mode == 'wtest' else [train_labels, val_labels]

            return images, labels

    def _get_files(self, format:str ) -> List[str]:
        """ Collects source files to be converted. """

        files = glob(os.path.join(self.src_dir, f'*.{format}' ))
        files = [file for file in files if "sample" not in file]

        # sanity check:
        already_patched = glob(os.path.join(self.dst_dir, f'*_?_?.{format}' ))
        if len(already_patched) > 0:
            print(f"Tiler: Warning: found tile annotations (e.g. {already_patched[0]}) in source folder. ")
        
        files = [file for file in files if file not in already_patched]

        return files


    def _split_images(self, 
                    image_list: list, 
                    mask_list: list = None, 
                    empty_images: list = None,
                    train_balance: bool = False) -> Tuple[List, List] :
        """ Splits images in train, val and test. 
            Returns: tuple containing lists of train, val, test images and masks"""

        n_images = len(image_list)
        n_val_imgs = round(self.ratio[1] * n_images)
        n_test_imgs = round(self.ratio[2] * n_images)

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

    def _check_already_splitted(self) -> bool:
        """ Checks if train, val, test folder are created and contain images"""

        skip_splitting = True
        no_folders = False
        empty_fold = False

        # check existance of dirs:
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'masks'] if self.task == 'segmentation' else ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                dir = os.path.join(self.dst_dir, self.task, self.image_type, subfold, subsubfold)
                if not os.path.isdir(dir):
                    no_folders = True
                    skip_splitting = False
                else:
                    files = [file for file in os.listdir(dir)]
                    if len(files) == 0:
                        empty_fold = True if subfold != 'test' else empty_fold
                        skip_splitting = False if subfold != 'test' else skip_splitting
        
        if skip_splitting is False:
            if no_folders is True:
                print("'train', 'val', 'test' folders not found. Deleting dataset and creating a new one.")
            elif empty_fold is True:
                print(f"No file found in {dir}.  Deleting existing dataset and creating a new one.")
        else:
            self.log.info("Dataset already splitted in train, val, test sets ✅. ")
        return skip_splitting
    
    def _make_folders(self) -> None:
        """ Clears the old dataset and creates a new one. """

        new_datafolder = os.path.join(self.dst_dir, self.task, self.image_type)
        if os.path.isdir(new_datafolder):
            shutil.rmtree(path = new_datafolder)
            print(f"Dataset at: {new_datafolder} removed.")
        
        # 1) makedirs:
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(new_datafolder, subfold, subsubfold), exist_ok=True)

        return new_datafolder
    
    def _move_files(self, images: list, labels: list, dataset_dir:str):

        subfolds_names = ['train', 'val', 'test']
        for img_dir, label_dir, fold in zip(images, labels, subfolds_names):
            for image in tqdm(img_dir, desc = f"Filling {fold}"):
                img_fn = os.path.split(image)[1]
                src_img, dst_img = image, os.path.join(dataset_dir, fold, 'images', img_fn)
                if self.safe_copy is True:
                    shutil.copy(src = src_img, dst = dst_img)
                else:
                    shutil.move(src = src_img, dst = dst_img)
            for label in label_dir:
                label_fn = os.path.split(label)[1]
                src_label, dst_label = label, os.path.join(dataset_dir, fold, 'labels', label_fn)
                dst_label = dst_label.replace('-labelled', '') # YOLO needs the same exact name between images and masks.
                if self.safe_copy is True:
                    shutil.copy(src = src_label, dst = dst_label)
                else:
                    shutil.move(src = src_img, dst = dst_img)

    def __call__(self):

        if self.task == 'detection' and self.image_type == 'wsi':
            self._split_yolo_wsi()
        elif self.task == 'detection' and self.image_type == 'tile':
            print(f"detection + tile for splitter is still to be tested")
            raise NotImplementedError()
            self._split_yolo_tiles()

        return




def test_Splitter():

    print(" ########################    TEST 1: ⏳    ########################")
    src_dir = '/Users/marco/Downloads/another_test'
    dst_dir = '/Users/marco/Downloads/boh'
    image_format = 'tif'
    ratio = [0.7, 0.3]
    task = 'detection'
    verbose = True
    safe_copy = True

    splitter = Splitter(src_dir=src_dir,
                        dst_dir=dst_dir,
                        image_format=image_format,
                        ratio=ratio,
                        task=task,
                        verbose = verbose, 
                        safe_copy = safe_copy)
    splitter()
    print(" ########################    TEST 1: ✅    ########################")

    # raise Exception

    print(f"Testing: dataset removed.")
    shutil.rmtree(path= os.path.join(dst_dir, task) )

    print(" ########################    TEST 2: ⏳     ########################")
    src_dir = '/Users/marco/Downloads/another_test'
    dst_dir = '/Users/marco/Downloads/boh'
    image_format = 'tif'
    ratio = [0.6, 0.3, 0.1]
    task = 'detection'
    verbose = True
    safe_copy = True

    splitter = Splitter(src_dir=src_dir,
                        dst_dir=dst_dir,
                        image_format=image_format,
                        ratio=ratio,
                        task=task,
                        verbose = verbose, 
                        safe_copy = safe_copy)
    splitter()
    print(" ########################    TEST 2: ✅    ########################")


    return


if __name__ == '__main__':
    test_Splitter()