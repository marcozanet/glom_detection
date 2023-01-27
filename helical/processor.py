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
import shutil
from tqdm import tqdm
from skimage import measure, io, color
import random
from loggers import get_logger


class Processor():

    def __init__(self,
                src_root: str, 
                dst_root: str, 
                ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                empty_perc: float = 0.1, 
                safe_copy = True) -> None:

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
        assert isinstance(safe_copy, bool), f"safe_copy should be boolean."

        self.src_root = src_root
        self.dst_root = dst_root
        self.ratio = ratio
        self.task = task 
        self.empty_perc = 0.1 if empty_perc is None else empty_perc
        self.safe_copy = safe_copy

        return


    def _get_images_masks(self) -> list:
        """ Collects images and masks from root folder. """

        # read through subfolds and collect images and masks:
        images = []
        for root, _, files in os.walk(self.src_root):
            imgs_found = [os.path.join(root, file) for file in files if 'png' in file or 'jpg' in file and 'DS' not in file]
            if isinstance(imgs_found, list):
                images.extend(imgs_found)
            elif isinstance(imgs_found, str):
                images.append(imgs_found)
        image_list = [image for image in images if 'labelled' not in image and 'mask' not in image]
        mask_list = []
        for i, image in enumerate(image_list):
            if os.path.isfile(image.replace('.png', '-labelled.png')):
                mask_list.append(image.replace('.png', '-labelled.png'))
            else:
                image_list.pop(i)

        # additional check
        additional_imgs = [(i, img) for (i, img) in enumerate(image_list) if img.replace('.png', '-labelled.png') not in mask_list]
        for i, _ in additional_imgs:
            image_list.pop(i)
        
        assert len(image_list) == len(mask_list), f"image_list has length {image_list} but mask_list has length {mask_list}."

        return image_list, mask_list

    
    def _get_images_labels(self) -> list:
        """ Collects images and labels from root folder. """

        # 1) read through subfolds and collect images and masks:
        images = []
        for root, _, files in os.walk(self.src_root):
            imgs_found = [os.path.join(root, file) for file in files if 'png' in file or 'jpg' in file and 'DS' not in file]
            if isinstance(imgs_found, list):
                images.extend(imgs_found)
            elif isinstance(imgs_found, str):
                images.append(imgs_found)
        image_list = [image for image in images if 'labelled' not in image and 'mask' not in image]

        # 2) get labels for images that have one (i.e. that have an obj)
        label_list = []
        images_unmatched = []
        return_images = []
        for _, image in enumerate(image_list):
            if os.path.isfile(image.replace('.png', '-labelled.txt')):      # 0.9x + 0.1x = x -> 0.9x = 70 -> x= 70/0.9
                label_list.append(image.replace('.png', '-labelled.txt'))
                return_images.append(image)
            else:
                images_unmatched.append(image)
        
        # 3) add a percentage of (randomly chosen) emtpy images:
        n_files = len(label_list) / (1 - self.empty_perc) # 0.9x + 0.1x = x -> 0.9x = 70 -> x= 70/0.9
        num_empty_images = int(self.empty_perc * n_files)

        assert len(images_unmatched) >= num_empty_images, f"images_unmatched has {len(images_unmatched)} elems but num_empty_images has {len(num_empty_images)}. "

        # 4) add (randomly chosen) empty images:
        rand_idxs = []
        while len(rand_idxs) < num_empty_images:
            idx = random.randint(0, len(images_unmatched)-1)
            if idx not in rand_idxs:
                rand_idxs.append(idx)
        empty_files = [images_unmatched[idx] for idx in rand_idxs]
        for file in empty_files:
            return_images.append(file)

        return return_images, images_unmatched


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
            
            assert mask_list is not None, ValueError(f"'mask_list' is None but should be provided.")

            train_masks = [image.replace('.png', '-labelled.png') for image in train_imgs if os.path.isfile(image.replace('.png', '-labelled.png'))]
            val_masks = [image.replace('.png', '-labelled.png') for image in val_imgs if os.path.isfile(image.replace('.png', '-labelled.png'))]
            test_masks = [image.replace('.png', '-labelled.png') for image in test_imgs if os.path.isfile(image.replace('.png', '-labelled.png'))]
            masks = [train_masks, val_masks, test_masks]

            assert len(train_imgs) == len(train_masks), f"train_images has {len(train_imgs)} images but train_masks has {len(train_masks)} masks. "
            assert len(val_imgs) == len(val_masks), f"val_images has {len(val_imgs)} images but val_masks has {len(val_masks)} masks. "
            assert len(test_imgs) == len(test_masks), f"test_images has {len(test_imgs)} images but test_masks has {len(test_masks)} masks. "
            
            return images, masks

        elif self.task == 'detection':

            assert empty_images is not None, ValueError(f"'empty_images' is None but should be provided.")
            train_labels = [image.replace('.png', '-labelled.txt') for image in train_imgs if os.path.isfile(image.replace('.png', '-labelled.txt'))]
            val_labels = [image.replace('.png', '-labelled.txt') for image in val_imgs if os.path.isfile(image.replace('.png', '-labelled.txt'))]
            test_labels = [image.replace('.png', '-labelled.txt') for image in test_imgs if os.path.isfile(image.replace('.png', '-labelled.txt'))]

            # now also add emtpy images to train
            n_tot_imgs =  len(train_imgs) / self.empty_perc
            n_empty_imgs = int(self.empty_perc * n_tot_imgs)
            additional_empty_imgs = empty_images[:n_empty_imgs]
            for image in additional_empty_imgs:
                train_imgs.append(image)

            labels = [train_labels, val_labels, test_labels]

            return images, labels


    def _makedirs_moveimages(self, images: list,  masks: list = None, labels: list = None) -> None:
        """ Makes dirs for train, val, test and moves images into the respective folds. """

        assert self.task in ['segmentation', 'detection'], ValueError(f"'task'={self.task}, but should be either 'segmentation' or 'detection'.")
        root = self.dst_root

        # 1) makedirs:
        subfolds_names = ['train', 'val', 'test']
        subsubfolds_names = ['images', 'masks'] if self.task == 'segmentation' else ['images', 'labels']
        for subfold in subfolds_names:
            for subsubfold in subsubfolds_names:
                os.makedirs(os.path.join(root, self.task, subfold, subsubfold), exist_ok=True)

        # 2) move images and masks
        if self.task == 'segmentation':
            assert masks is not None, ValueError(f"'masks' is None, but 'task' is 'segmentation'.")
            
            for img_dir, mask_dir, fold in zip(images, masks, subfolds_names):
                for image, mask in tqdm(zip(img_dir, mask_dir), total= len(img_dir), desc= f"Filling {fold} dir"):
                    img_fn = os.path.split(image)[1]
                    mask_fn = os.path.split(mask)[1]
                    src_img, dst_img = image, os.path.join(root, self.task, fold, 'images', img_fn)
                    src_mask, dst_mask = mask, os.path.join(root, self.task, fold, 'masks', mask_fn)
                    dst_mask = dst_mask.replace('-labelled', '') # YOLO needs the same exact name between images and masks.
                    if self.safe_copy is True:
                        shutil.copy(src = src_img, dst = dst_img )
                        shutil.copy(src = src_mask, dst = dst_mask)
                    else:
                        shutil.move(src = src_img, dst = dst_img )
                        shutil.move(src = src_mask, dst = dst_mask)

        elif self.task == 'detection':
            assert labels is not None, ValueError(f"'labels' is None, but 'task' is 'detection'.")

            for img_dir, label_dir, fold in zip(images, labels, subfolds_names):
                for image in tqdm(img_dir, desc = f"Filling {fold}"):
                    img_fn = os.path.split(image)[1]
                    src_img, dst_img = image, os.path.join(root, self.task, fold, 'images', img_fn)
                    if self.safe_copy is True:
                        shutil.copy(src = src_img, dst = dst_img)
                    else:
                        shutil.move(src = src_img, dst = dst_img)
                for label in label_dir:
                    label_fn = os.path.split(label)[1]
                    src_label, dst_label = label, os.path.join(root, self.task, fold, 'labels', label_fn)
                    dst_label = dst_label.replace('-labelled', '') # YOLO needs the same exact name between images and masks.
                    if self.safe_copy is True:
                        shutil.copy(src = src_label, dst = dst_label)
                    else:
                        shutil.move(src = src_img, dst = dst_img)

                
        return
    

    def get_trainvaltest(self) :

        traindir = os.path.join(self.dst_root, self.task, 'train')
        valdir = os.path.join(self.dst_root, self.task, 'val')
        testdir =  os.path.join(self.dst_root, self.task, 'test')

        # 1) check if dataset already exists:
        skip_splitting = self._check_already_splitted()
        if skip_splitting:
            self.log.info("Splitting in train, val, test sets ✅. ")
            return traindir, valdir, testdir
        else:
            self._clear_dataset()

        if self.task == 'segmentation':
            
            image_list, mask_list = self._get_images_masks()
            images, masks = self._split_images_trainvaltest(image_list = image_list, mask_list= mask_list)
            self._makedirs_moveimages(images=images, masks=masks)
            
        elif self.task == 'detection':

            image_list, empty_list = self._get_images_labels()
            fullimgs_list = [file for file in image_list if file not in empty_list]
            images, labels = self._split_images_trainvaltest(image_list = fullimgs_list, empty_images= empty_list)
            self._makedirs_moveimages(images=images, labels=labels)


        assert os.path.isdir(traindir), f"{traindir} is not a dir."
        assert os.path.isdir(valdir), f"{valdir} is not a dir."
        assert os.path.isdir(testdir), f"{testdir} is not a dir."

        self.log.info("Splitting in train, val, test sets ✅. ")

        return traindir, valdir, testdir


    def make_bboxes(self, 
                    clear_txt: bool = False, 
                    reduce_classes: bool = False)-> None:
        """ Converts mask images from a folder to bounding boxes and saves them in the same folder. 
            Masks MUST be named like: <name-labelled.png>
            clear_txt = if True, all txt files are removed from src_root before making txt_files.
            reduce_classes = if True, it makes unhealthy class = NA class -> classes = {0: healthy, 1: unhealthy}. 
                             if False, classes are kept as original -> classes = {0: healthy, 1: NA, 2:unhealthy}"""
        
        folder = self.src_root
        assert os.path.isdir(folder), ValueError(f"'folder': {folder} is not a dir.")

        # 1) delete all previous txt bbox files:
        def _del_all_txtfiles(root: str):
            """ Deletes all txt files from folder and subfolders. """

            # 1.1) collect txt files
            all_txt = []
            for root, _, files in os.walk(root):
                txt_files = [os.path.join(root, file) for file in files if '.txt' in file]
                if isinstance(txt_files, list):
                    all_txt.extend(txt_files)
                elif isinstance(txt_files, str):
                    all_txt.append(txt_files)
            # 1.2) remove txt files:
            for txt in tqdm(all_txt, desc=f"Clearing txt bbox"):
                os.remove(txt)

            return

        if clear_txt is True:
            _del_all_txtfiles(folder)

        # 2) collect masks:
        masks = []
        for root, _, files in os.walk(folder):
            masks_found = [os.path.join(root, file) for file in files if 'DS' not in file and '-labelled.png' in file]
            if isinstance(masks_found, list):
                masks.extend(masks_found)
            elif isinstance(masks_found, str):
                masks.append(masks_found)
        masks = [file for file in masks if isinstance(file, str)] # additional filter check

        # 3) convert masks:
        def _convert_mask2bb(file, ignore_area = 1000):
            """ Converts ONE mask to bb. """

            # 3.1) label different objects differently:
            image = io.imread(file)
            W, H = image.shape[:2]
            image = color.rgba2rgb(image) if image.ndim == 3 else np.expand_dims(image, axis = 2)
            rgb_image = image
            image = color.rgb2gray(image) * 255 if image.shape[2] == 3 else image
            image = np.uint8(image)
            image = np.where(image == 255, 0, image) # black = tissue, white = background (no objs)
            labels, num_objs = measure.label(image, return_num= True)

            # 3.2) get class, xc, yc, w, h for each object
            props = measure.regionprops(labels)
            text = ''
            n_healthy = 0
            n_unhealthy = 0
            n_na = 0
            for i in range(0, num_objs):
                filetxt = file.replace('.png', '.txt')
                if os.path.isfile(filetxt) or props[i]['area'] < ignore_area: # if already computed or min area to be considered obj
                    continue
                y_min, x_min, y_max, x_max = props[i]['bbox']
                xc, yc = (x_min + x_max)//2, (y_min + y_max)//2
                w, h = x_max - x_min, y_max - y_min
                # normalize:
                xc, w = xc/W, w/W
                yc, h = yc/H, h/H
                # assign a class
                classes = {(0.,1.,0.):0, (0.,0.,1.):1, (1., 0., 0.):2} if reduce_classes is False else {(0.,1.,0.):0, (0.,0.,1.):0, (1., 0., 0.):1 }
                cls_xy = tuple(np.argwhere(labels == i+1)[0])
                cls = classes[tuple(rgb_image[cls_xy])]
                text += f"\n{cls} {xc} {yc} {w} {h}" 

                # count pos and neg instances for class imbalance (to be used in get_trainvaldir())
                if reduce_classes is True:
                    if cls == 0:
                        n_healthy += 1
                    elif cls == 1:
                        n_unhealthy += 1
                else:
                    if cls == 0:
                        n_healthy += 1
                    elif cls == 1:
                        n_na += 1   
                    elif cls == 2:
                        n_unhealthy += 1   
    
            # 3.3) save to txt
            if num_objs > 0: # for emtpy images no txt needed
                with open(filetxt, 'w') as f:
                    f.write(text)

            if reduce_classes is True:
                mask_instances = {'n_healthy': n_healthy, 'n_unhealthy': n_unhealthy}
            else:
                mask_instances = {'n_healthy': n_healthy, 'n_unhealthy': n_unhealthy, 'n_na': n_na}

            return mask_instances

        count_classes = {}
        for mask in tqdm(masks, desc = f"Converting masks to bboxes"):
            fname = os.path.split(mask)[1]
            mask_instances = _convert_mask2bb(mask)
            count_classes[fname] = mask_instances

        return 
    

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
                dir = os.path.join(self.dst_root, self.task, subfold, subsubfold)
                if not os.path.isdir(dir):
                    no_folders = True
                    skip_splitting = False
                else:
                    files = [file for file in os.listdir(dir) if 'png' in file or 'txt' in file]
                    if len(files) == 0:
                        empty_fold = True
                        skip_splitting = False
        
        if skip_splitting is False:
            if no_folders is True:
                print("'train', 'val', 'test' folders not found. Deleting dataset and creating a new one.")
            elif empty_fold is True:
                print(f" No file found in {dir}.  Deleting existing dataset and creating a new one.")


        return skip_splitting
    

    def _clear_dataset(self) -> None:
        """ Clears the old dataset and creates a new one. """

        del_dir = os.path.join(self.dst_root, self.task)
        if os.path.isdir(del_dir):
            shutil.rmtree(path = del_dir)
            print(f"Dataset at: {del_dir} removed.")
        
        os.makedirs(del_dir)

        return


