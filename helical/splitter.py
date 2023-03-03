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
                 task = Literal['detection', 'segmentation', 'MIL', 'mil', 'both'],
                 safe_copy:bool = True,
                 verbose:bool = False) -> None:

        self.log = get_logger()

        ALLOWED_IMAGE_FORMATS = ['tif', 'tiff', 'png']
        ALLOWED_LABEL_FORMATS = ['txt', 'gson', 'json']
        ALLOWED_TASKS = ['detection', 'segmentation', 'MIL', 'mil', 'both']

        assert image_format in ALLOWED_IMAGE_FORMATS, f"'image_format':{image_format} should be one of {ALLOWED_IMAGE_FORMATS}"
        # assert label_format in ALLOWED_LABEL_FORMATS, f"'label_format':{label_format} should be one of {ALLOWED_LABEL_FORMATS}"
        assert os.path.isdir(dst_dir), f"'dst_dir':{dst_dir} is not a valid filepath."
        assert os.path.isdir(src_dir), f"'src_dir':{src_dir} is not a valid filepath."
        assert task in ALLOWED_TASKS, ValueError(f"'task'= {task} should be one of {ALLOWED_TASKS}. ")
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

        return images
    
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

            self.log.info(f"{self.__class__.__name__}.{'_move_files'}: {fold}set: {[os.path.basename(file) for file in img_dir]}")

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
                    shutil.move(src = src_label, dst = dst_label)
    


    def move_already_tiled(self, tile_root:str):
        """ Moves tiles already tiled from root/images and root/labels to the 
            new_dataset and splits them based on the wsi splitting."""
        assert os.path.isdir(tile_root), f"'tile_root':{tile_root} is not a valid filepath."

        print(f"Moving already tiled images:")

        # 1) get slides fnames and folders:
        slides = []
        for root, dirs, files in os.walk(os.path.join(self.src_dir, self.task)):
            for file in files:
                if 'tif' in file:
                    slides.append(os.path.join(root, file))
        
        slides = [(os.path.dirname(slide), os.path.basename(slide).split('.')[0]) for slide in slides]

        # 2) make new folders for tiles:
        new_datafolder = os.path.join(self.dst_dir, self.task, 'tiles')
        if os.path.isdir(new_datafolder):
            shutil.rmtree(path = new_datafolder)
            print(f"Dataset at: {new_datafolder} removed.")
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(new_datafolder, subfold, subsubfold), exist_ok=True)

        # 3) for each tile, find its corresponding folder and move/copy it there:
        tile_images_fold = os.path.join(tile_root, 'images')
        n_moved_images = 0
        for tile_fn in tqdm(os.listdir(tile_images_fold), desc = "Moving images"):
            src_fp = os.path.join(tile_root, 'images', tile_fn)
            for slide_folder, slide_fn in slides:
                if slide_fn in src_fp:
                    dst_fp = os.path.join(slide_folder, tile_fn).replace('wsi', 'tiles')
                    if not os.path.isfile(dst_fp):
                        shutil.copy(src = src_fp, dst = dst_fp)
                        n_moved_images+= 1

        
        # 4) do the same for labels:
        n_moved_labels = 0
        tile_labels_fold = os.path.join(tile_root, 'labels')
        for tile_fn in tqdm(os.listdir(tile_labels_fold), desc = "Moving labels"):
            src_fp = os.path.join(tile_root, 'labels', tile_fn)
            for slide_folder, slide_fn in slides:
                if slide_fn in src_fp:
                    dst_fp = os.path.join(slide_folder, tile_fn).replace('wsi', 'tiles').replace('images', 'labels')
                    # print(src_fp, dst_fp)
                    if not os.path.isfile(dst_fp):
                        shutil.copy(src = src_fp, dst = dst_fp)
                        n_moved_labels+= 1

        print(f"Moved {n_moved_images} images from {len(os.listdir(tile_images_fold))} original images")
        print(f"Moved {n_moved_labels} labels from {len(os.listdir(tile_labels_fold))} original labels ")

        return
    
    def _remove_empty_images(self):

        # print(os.path.join(dst_dir, task, 'tiles', '*', 'labels', '*.txt' ))
        # 1) get all image tiles from the train, val, test
        train_images = glob(os.path.join(self.dst_dir, self.task, 'tiles', 'train', 'images', '*.png' ))
        assert all(['wsi' not in file for file in train_images]), f"All selected files should be tile images, not wsi images."
        assert all(['images' in file for file in train_images]), f"All selected files should be tile images, not labels."
        assert len(train_images)> 0, f"No image found."
        # 2) get all label tiles from train, val, test
        train_labels = glob(os.path.join(self.dst_dir, self.task, 'tiles', 'train', 'labels', '*.txt' ))
        train_labels = [label for label in train_labels if 'test' not in train_labels]
        assert all(['wsi' not in file for file in train_labels]), f"All selected files should be tile labels, not wsi labels."
        assert all(['labels' in file for file in train_labels]), f"All selected files should be tile labels, not images."
        assert len(train_labels)> 0, f"No label found."

        # 3) collect images without a label (i.e. empty)
        train_empty = [image for image in train_images if image.replace('images', 'labels').replace('png', 'txt') not in train_labels]
        assert all([not os.path.isfile(image.replace('images', 'labels').replace('png', 'txt')) for image in train_empty])

        # 4) from this files keep an empty_perc and delete the other ones:
        train_full = [image for image in train_images if image.replace('images', 'labels').replace('png', 'txt') in train_labels]
        # train_images, trin_labels = [image for image in images if 'val' in image], [label for label in labels if 'val' in labels]
        
        # check if empty percantage already <= empty_perc:
        if (len(train_empty)/len(train_images)) <= self.empty_perc:
            print(f"Empty perc of images: {round(len(train_empty)/len(train_images), 2)} already <= {self.empty_perc}. ")
            return

        # delete random empty images:
        # print(f"train images: {len(train_images)}")
        # print(f'train_full: {len(train_full)}')
        # print(f"train empty: {len(train_empty)}")
        n_train_wished =  int(len(train_full) * (1+ self.empty_perc))
        n_del_train =  len(train_images) - n_train_wished 
        train_del_imgs = random.sample(train_empty, n_del_train )
        assert all([not os.path.isfile(image.replace('images', 'labels').replace('png', 'txt')) for image in train_del_imgs])
        for file in tqdm(train_del_imgs):
            os.remove(file)
        final_train_images = glob(os.path.join(self.dst_dir, self.task, 'tiles', 'train', 'images', '*.png' ))
        print(f"Removed {n_del_train} empty images. Train images: {len(train_images)} -> {len(final_train_images)} .")
        assert len(final_train_images) == n_train_wished, f"Wished: {n_train_wished}, obtained: {len(final_train_images)}"


        return


    def __call__(self):

        if self.task == 'detection' and self.image_type == 'wsi':
            self._split_yolo_wsi()
        elif self.task == 'detection' and self.image_type == 'tile':
            print(f"detection + tile for splitter is still to be tested")
            raise NotImplementedError()
            self._split_yolo_tiles()

        return




def test_Splitter():

    # print(" ########################    TEST 1: ⏳    ########################")
    # src_dir = '/Users/marco/Downloads/another_test'
    # dst_dir = '/Users/marco/Downloads/boh'
    # image_format = 'tif'
    # ratio = [0.7, 0.3]
    # task = 'detection'
    # verbose = True
    # safe_copy = True

    # splitter = Splitter(src_dir=src_dir,
    #                     dst_dir=dst_dir,
    #                     image_format=image_format,
    #                     ratio=ratio,
    #                     task=task,
    #                     verbose = verbose, 
    #                     safe_copy = safe_copy)
    # splitter()
    # print(" ########################    TEST 1: ✅    ########################")

    # raise Exception

    # print(f"Testing: dataset removed.")
    # shutil.rmtree(path= os.path.join(dst_dir, task) )

    print(" ########################    TEST 2: ⏳     ########################")
    src_dir = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    dst_dir = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    image_format = 'tif'
    ratio = [0.6, 0.2, 0.2]
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
    # splitter.move_already_tiled(tile_root = '/Users/marco/Downloads/muw_slides')
    # splitter._remove_empty_images()
    print(" ########################    TEST 2: ✅    ########################")


    # print(" ########################    TEST 3: ⏳     ########################")
    # src_dir = '/Users/marco/Downloads/test_folders/test_featureextractor/images'
    # dst_dir = '/Users/marco/Downloads/test_folders/test_milsplitter'
    # image_format = 'png'
    # ratio = [0.6, 0.2, 0.2]
    # task = 'detection'
    # verbose = True
    # safe_copy = True

    # splitter = Splitter(src_dir=src_dir,
    #                     dst_dir=dst_dir,
    #                     image_format=image_format,
    #                     ratio=ratio,
    #                     task=task,
    #                     verbose = verbose, 
    #                     safe_copy = safe_copy)
    # splitter()
    # # splitter.move_already_tiled(tile_root = '/Users/marco/Downloads/muw_slides')
    # # splitter._remove_empty_images()
    # print(" ########################    TEST 3: ✅    ########################")

    

    return


if __name__ == '__main__':
    test_Splitter()