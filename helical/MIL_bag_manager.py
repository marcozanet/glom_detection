import os
from glob import glob
from tqdm import tqdm
import albumentations as A
import random
import cv2
import numpy as np
from MIL_bags_creation_new import BagCreator
from cnn_feat_extract_main import extract_cnn_features


class BagManager():

    def __init__(self,
                 folder:str,
                 all_slides_dir:str,
                 map_classes:dict, # e.g. {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2},
                 bag_classes:dict, # e.g. {0:0.25, 1:0.5, 2:0.75, 3:1},
                 stain:str='pas',
                 img_fmt:str = '.jpg', 
                 feat_fmt:str = '.npy',
                 n_instances_per_bag:int=9 ) -> None:
        

        self.stain = stain.upper()
        self.folder = folder 
        self.all_slides_dir = all_slides_dir
        self.map_classes = map_classes
        self.n_instances_per_bag = n_instances_per_bag
        self.bag_classes = bag_classes
        self._images_path_like = os.path.join(folder, '*', '*.jpg')
        self.all_images = glob(self._images_path_like)
        self.img_fmt = img_fmt
        self.feat_fmt = feat_fmt
        self.transform = A.Compose([A.OneOf([A.ToGray(p=0.2),
                                    A.CLAHE(p=0.5),
                                    A.RandomBrightnessContrast(p=0.2)]),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5)])
        self._parse_()

        return
    

    def _parse_(self):
        """ Parses class args. """

        assert self.stain in ['PAS'], f"Not implemented for stains other than PAS."
        assert os.path.isdir(self.folder), f"'folder':{self.folder} is not a valid dirpath."
        assert os.path.isdir(self.all_slides_dir), f"'folder':{self.all_slides_dir} is not a valid dirpath."
        assert isinstance(self.map_classes, dict), f"'map_classes:{self.map_classes} is type {type(self.map_classes)}, but should be a dict."
        assert isinstance(self.bag_classes, dict), f"'bag_classes:{self.bag_classes} is type {type(self.bag_classes)}, but should be a dict."
        # check that 'bag_classes' is zero indexed: 
        assert list(self.bag_classes.keys())[0] == 0, f"ZeroIndexError: 'bag_classes':{self.bag_classes} should be zero indexed."
        assert len(self.all_images)>0, f"No images like: {self._images_path_like}"
        
        return
        
    
    def _del_augm_files(self):
        """ Deletes """
        
        del_files = [file for file in self.all_images if 'Augm' in file]
        for file in del_files:
            os.remove(file)
        print(f"Deleted all augmented files.")
        self.all_images = glob(self._images_path_like)

        return
    
        
        


    def _augment_wsi(self, n_augm_files:int, basename:str, wsi_files:list):
        """ Augment slide so that it has at least 9 instances to use to create bags. 
            Uses false pos files from wsi (if available) or from other wsis and augment 
            them until there's enough images. It happens e.g. that only a few 
            gloms/fp are detected by YOLO and that's not enough to make a bag."""
        
        # helper funcs:
        change_name_augm = lambda fp, i: os.path.join(os.path.dirname(fp), f"Augm_{i}_{os.path.basename(fp)}")
        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]

        # augmenting false pos (not gloms):
        # print(f"Creating {n_augm_files} files for {basename} slide.")
        already_used = []
        files_exist = False
        for i in range(n_augm_files): 
            false_pos_files = [file for file in wsi_files if 'false_positives' in file and file not in already_used and 'Augm' not in file and not os.path.isfile(change_name_augm(file, i))]
            if len(false_pos_files)==0: # then use false pos from other files
                false_pos_files = [file for file in self.all_images if 'false_positives' in file and file not in already_used and 'Augm' not in file and not os.path.isfile(change_name_augm(file, i))]
                assert len(false_pos_files)>0
            false_pos_file = random.choice(false_pos_files) # pick a random false pos
            img = cv2.imread(false_pos_file, cv2.COLOR_BGR2RGB)
            transformed_img = self.transform(image=img)['image'] # augment
            false_pos_file = false_pos_file.replace(get_wsi_fname(false_pos_file), basename) # fake new name for wsi that didn't have fals pos
            write_fp = change_name_augm(false_pos_file, i)
            if not os.path.isfile(write_fp): # save
                cv2.imwrite(write_fp, transformed_img)
                already_used.append(false_pos_file)
            else:
                files_exist=True
        
        if files_exist:
            print(f"Some files already existing.")
        assert n_augm_files == len(already_used)

        return
    
    
    def _augment_false_positives(self):

        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]
        print(get_wsi_fname(self.all_images[0]))
        basenames = list(set([get_wsi_fname(file) for file in self.all_images]))
        assert len(basenames)>0, f"No file in 'basenames'"
        for basename in tqdm(basenames, f"Augmenting FP to have min {self.n_instances_per_bag} images per bag"): 
            wsi_files = [file for file in self.all_images if basename in file]
            if len(wsi_files)<self.n_instances_per_bag: 
                n_augm_files = self.n_instances_per_bag - len(wsi_files)
                self._augment_wsi(n_augm_files, basename, wsi_files)

        return
    
    def create_bags(self):

        # instances_folder = '/Users/marco/helical_tests/test_cnn_zaneta/cnn_dataset'
        # map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2}
        # sclerosed_idx=2
        # n_instances_per_bag = 9
        # n_classes = 4
        # all_slides_dir='/Users/marco/Downloads/zaneta_files/safe'

        creator = BagCreator(instances_folder=self.folder, 
                            map_classes=self.map_classes,
                            instance_fmt = self.img_fmt, # here feat extraction not happened yet
                            n_instances_per_bag=self.n_instances_per_bag,
                            all_slides_dir=self.all_slides_dir,
                            img_fmt=self.img_fmt)
        creator()

        self.bags_indices = creator.bags_indices
        # self.file_labels = creator.file_labels
        self.bags_labels = creator.bags_labels
        # print(creator.n_classes)
        # self.n_classes = creator.n_classes
        self.slides_labels = creator.slides_labels
        self.instances_idcs = creator.instances_idcs

        self._get_class_frequency()

        

        return

    def _get_class_frequency(self):
        """Gets labels frequency per bag. """

        # print(self.bags_labels)
        # print(self.bag_classes)
        frequency = {k:0 for k in self.bag_classes.keys()}
        # print(frequency)
        for bag_idx, bag_label in self.bags_labels.items():
            for clss, thres in self.bag_classes.items():
                if bag_label == clss:
                    frequency[clss]+=1 
        
        assert np.array(list(frequency.values())).sum() == len(self.bags_indices), f"{np.array(list(frequency.values())).max()} != {len(self.bags_indices)}"
        print(f"Bag classes: {frequency}")
        self.class_freq = frequency

        return
    
    def augment2createbags(self):
        """ Augment data to have at least 9 instances per bag. """

        self._augment_false_positives()

        return

    def _augment_bags(self):
        
        # get max num of classes
        n_classes = np.array(list(self.class_freq.values()))
        n_maxclass = ((n_classes.argmax()), n_classes.max())
        n_classes = [(clss, n) for clss, n in enumerate(n_classes) ]
        bags2create = {clss:(n_maxclass[1] - n) for clss, n in n_classes if n !=n_maxclass[1]}

        # for each class to be augmented: 
        matchingslides = lambda slides_labels, class2match: [slide for slide, clss in slides_labels.items() if class2match == clss]
        new_bags = {}
        bag_n = len(self.bags_indices) 
        augm_iter = 0 # n of iterations needed to complete balancing
        for clss, needed_bags in tqdm(bags2create.items(), desc='Balancing classes with augmentation'):
            created_bags = 0
            # get ROI/ images that do belong to that class 
            class_slides = matchingslides(slides_labels=self.slides_labels, class2match=clss)
            # retrieve all images of that WSI: 
            class_images = [file for file in self.all_images if any([slide in file for slide in class_slides])]
            # for each WSI available create a number of bags: 
            while created_bags != needed_bags:
                augm_iter+=1
                for class_slide in class_slides:
                    # until all needed bags are done or until all images of that slide are used:
                    remaining = [file for file in class_images if class_slide in file and 'Augm' not in file]
                    while created_bags != needed_bags and len(remaining) >= self.n_instances_per_bag:
                        # check that these images have at least 1 non false pos: TODO
                        # create the new bags:
                        selected = random.sample(remaining, k = self.n_instances_per_bag)
                        remaining = [file for file in remaining if file not in selected ]
                        self._augment_selected(selected=selected, augm_iter=augm_iter)
                        new_bag = {self.instances_idcs[file]: file for file in selected}
                        created_bags += 1
                        bag_n += 1
                        new_bags.update({bag_n: new_bag})
        


        self.bags_indices.update(new_bags)
        print(f"Created {len(self.bags_indices)} bags. ")

        # update labels: 
        self.get_bags_labels()
        # update class frequency: 
        self._get_class_frequency()

        return
    

    def get_bags_labels(self) -> None: 

        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]
        
        bags_labels = {}
        for bag_idx, bag in tqdm(self.bags_indices.items(), desc='Assigning labels to bags'):
            slide_name = get_wsi_fname(list(bag.values())[0])
            assert all([slide_name == get_wsi_fname(val) for val in bag.values()])
            bag_label = {bag_idx:self.slides_labels[slide_name]}
            bags_labels.update(bag_label)
        self.bags_labels = bags_labels

        return 
    

    def _augment_selected(self, selected:list, augm_iter:int):

        change_name_augm = lambda fp, i: os.path.join(os.path.dirname(fp), f"Augm_{i}_{os.path.basename(fp)}")

        already_existing = False
        for file in selected:
            img = cv2.imread(file, cv2.COLOR_BGR2RGB)
            transformed_img = self.transform(image=img)['image'] # augment
            write_fp = change_name_augm(file, augm_iter)
            if not os.path.isfile(write_fp): # save
                cv2.imwrite(write_fp, transformed_img)
            else:
                already_existing=True
        if already_existing:        
            print(f"Some augmented files exist already.")
        
        n_instances_idcs = len(self.instances_idcs)
        self.instances_idcs.update({(n_instances_idcs+k):file for k, file in enumerate(selected)})

        return
    
    
    def _final_check(self)->None:
        """ Checks the final created bags (balancing, min instances per bag..)"""

        for bag_idx, bag in self.bags_indices.items():
            # 1) check that each bag has at least 9 el  
            assert len(bag) >= self.n_instances_per_bag, f"Length of bag is {len(bag)} but min should be {self.n_instances_per_bag} "
            # 2) check that files in bags exist
            assert all([os.path.isfile(file) for file in list(bag.values()) ])

        # 3) check that class bags are balanced 
        assert all([list(self.class_freq.values())[0] == v  for k, v in self.class_freq.items()]), f"Classes are not balanced: values are not all the same in {self.class_freq}"

        return
    
    def extract_features(self)->None:
        """ Extracts features on the specified folder."""
        
        print('*'*20)
        print(f"Extracting features'")
        print('*'*20)
        extract_cnn_features()

        return
    

    def convert_imagebags2featbags(self):

        # img2feat = lambda img_fp: os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_fp)))), 'feat_extract', os.path.dirname(os.path.dirname(img_fp)), os.path.dirname(img_fp), os.path.basename(img_fp).replace(self.img_fmt, self.feat_fmt) )
        img2feat = lambda img_fp: os.path.join(os.path.dirname( os.path.dirname(    os.path.dirname(    os.path.dirname(img_fp) ) )), 'feat_extract', os.path.split(os.path.dirname(os.path.dirname(img_fp)))[-1], os.path.split(os.path.dirname(img_fp))[-1], 'feats', os.path.basename(img_fp).replace(self.img_fmt, self.feat_fmt))

        for bag_idx, bag in self.bags_indices.items():
            for img_idx, img_fp in bag.items():
                feat_fp = img2feat(img_fp)
                # print(feat_fp)
                # assert os.path.isfile(feat_fp), f"Feat corresponding to image {os.path.basename(img_fp)} does not exist. "
                self.bags_indices[bag_idx][img_idx] = feat_fp
        
        print(self.bags_indices)

        self._final_check()

        return



    def __call__(self) -> None:
        
        print('*'*20)
        print(f"Creating Bags from '{os.path.join('/'+os.path.split(os.path.split(self.folder)[0])[-1], os.path.split(self.folder)[-1])}'")
        print('*'*20)
        # self._del_augm_files()
        files = glob(self._images_path_like)
        print(f"Initial files: {len(files)}")

        self.augment2createbags()
        files = glob(self._images_path_like)
        print(f"Files after FP augmentation: {len(files)}")

        self.create_bags()
        self._augment_bags()
        self._final_check()
        files = glob(self._images_path_like)
        print(f"Files after class balancing: {len(files)}")

        # self.convert_imagebags2featbags()

        return self.bags_indices, self.bags_labels

 


if __name__ == '__main__':

    folder = '/Users/marco/helical_tests/test_cnn_zaneta/cnn_dataset/train'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2}
    bag_classes = {0:0.25, 1:0.5, 2:0.75, 3:1}
    stain = 'PAS'
    n_instances_per_bag=9
    all_slides_dir='/Users/marco/Downloads/zaneta_files/safe'
    train_bag_manager = BagManager(folder=folder, 
                          map_classes=map_classes,
                          bag_classes=bag_classes,
                          all_slides_dir=all_slides_dir,
                          stain=stain, 
                          n_instances_per_bag=n_instances_per_bag)
    train_bag_manager._del_augm_files()
    train_bag_manager()
    folder = folder.replace('train', 'test')
    test_bag_manager = BagManager(folder=folder, 
                          map_classes=map_classes,
                          bag_classes=bag_classes,
                          all_slides_dir=all_slides_dir,
                          stain=stain, 
                          n_instances_per_bag=n_instances_per_bag)
    test_bag_manager()
    # augmentor.