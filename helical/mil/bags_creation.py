import os 
from typing import Tuple, List
from glob import glob

import numpy as np
import random
from configurator import Configurator
import math

class BagCreator(Configurator): 
    def __init__(self, 
                instances_folder:str, 
                exp_folder:str,
                sclerosed_idx: int,
                n_images_per_bag: int,
                n_classes: int
                
                ) -> None:
        
        """ Given a root of instances and labels, 
        instances = folder with 2048x2048 tiles, labels = 0 if no SCLEROSED gloms within tile (i.e. tissue or healthy glom or glom NA).
        BagCreator will create 1 bag = 1 ROI of 4096x4096 -> the region is represented by the 9 tiles that fall within that region ."""
        
        super().__init__()

        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."       
       
        self.instances_folders = instances_folder
        self.sclerosed_idx = sclerosed_idx
        self.exp_folder = exp_folder
        self.class_name = self.__class__.__name__
        self.n_images_per_bag = n_images_per_bag
        self.n_classes = n_classes

        self.instances_idcs = self.get_instances_indices()

        return
    
    
    def get_instances_indices(self) -> None: 
        images = glob(os.path.join(self.instances_folders, "*.png"))
        # print(f"Total images: {len(images)}. Images per folder: {self.n_images_per_bag}, n_bags: {math.ceil(len(images)/self.n_images_per_bag)}. Repeating {len(images)%self.n_images_per_bag} to fill last folder")
        self.log.info(f"{self.class_name}.{'get_instances_indices'}: Total images: {len(images)}. Images per folder: {self.n_images_per_bag}, n_bags: {math.ceil(len(images)/self.n_images_per_bag)}. Repeating {len(images)%self.n_images_per_bag} to fill last folder")

        self.n_bags = math.ceil(len(images)/self.n_images_per_bag)
        
        instances_idcs = {fp:i for (i,fp) in enumerate(images) }
        return instances_idcs



    def get_bags_indices(self) -> dict: 

        images = glob(os.path.join(self.instances_folders, "*.png"))

        # get #samples per slide: 
        wsi_fnames = list(set([os.path.basename(file).split('_SFOG')[0] for file in images]))
        # print(wsi_fnames)
        wsi_samples = {}
        for wsi in wsi_fnames:
            samples = [int(file.split('sample')[1].split('_')[0]) for file in images if wsi in file]
            wsi_samples[wsi] = np.array(samples).max() + 1
        # print(wsi_samples)

        # create bags
        somma = 0
        bags = {}

        bag_n = 0
        for wsi, n_samples in wsi_samples.items():
            for i in range(n_samples):
                files = sorted([os.path.basename(file) for file in images if f"{wsi}_SFOG_sample{i}" in file])
                # print(f"files: {(files)}")
                somma += len(files)

                # fill bags:
                n_bags = len(files) // self.n_images_per_bag
                remaining = [os.path.join(self.instances_folders, file) for file in files]
                for _ in range(n_bags):
                    selected = random.sample(remaining, k = self.n_images_per_bag)
                    remaining = [file for file in remaining if file not in selected ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in selected}
                    # bags[bag_n] = [(self.instances_idcs[file],file) for file in selected]
                    bag_n += 1
                if len(remaining) > 0:
                    choosable = [file for file in files if file not in remaining]
                    remaining_bag = remaining + random.sample(choosable, k= self.n_images_per_bag-len(remaining) )
                    remaining_bag = [os.path.join(self.instances_folders, file) for file in remaining_bag ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in remaining_bag}
                
        # print(f"Bags indices:{bags}")
        # print(f"Bag[0] indices:{bags[0]}")
        self.log.info(f"{self.class_name}.{'get_bags_indices'}: Bag indices:{bags}")
        self.log.info(f"{self.class_name}.{'get_bags_indices'}: Bag[0] indices:{bags[0]}")
        return  bags
    

    # def get_bags_labels(self, bags: dict):

    #     bags_labels = {}
    #     for idx_bag, instances in bags.items():
    #         for idx_inst, instance in instances.items(): 
    #             label_fp= instance.replace('images', 'labels')

    #             # if label doesn't exist, then no sclerosed glom.
    #             if not os.path.isfile(label_fp):
    #                 bags_labels[idx_bag] = 0 
    #                 continue
                
    #             # open correspondent label:
    #             with open(label_fp, 'r') as f:
    #                 text = f.readlines()

    #             # bag label = 1 if at least one instance label = 1:
    #             label = 0
    #             for line in text:
    #                 clss = line.split(' ')[0]
    #                 if clss == self.sclerosed_idx:
    #                     label = 1 # i.e.
    #             bags_labels[idx_bag] = label

    #     # print(bags_labels)
        
    #     return  bags_labels

    def get_fake_labels(self, bags_indices:dict): 

        bags_labels = {idx_bag:random.randint(0,self.n_classes-1) for idx_bag in bags_indices.keys()}
        # print(f"bag_labels:{bag_labels}")
        self.log.info(f"{self.class_name}.{'get_fake_labels'}: n_classes:{self.n_classes}")
        self.log.info(f"{self.class_name}.{'get_fake_labels'}: bag_labels:{bags_labels}")

        return bags_labels

    
    def _get_bags_features(self, bags_indices:dict): 
        """ Returns bag features. """

        image2feat = lambda img_file: os.path.join(self.exp_folder, os.path.basename(img_file).replace('.png', ''),  'stage23_C3_features.npy')
        bags_features = {}
        for idx_bag, bag in bags_indices.items():
            # print(bag)
            # print(image2feat('/Users/marco/helical_tests/test_bagcreator/images/200701099_09_SFOG_sample0_0_4.png'))
            # bags_idcs like {0: {0: '....png', 6: '..png', 16:'..png'}, 1:{19: '....png', 1: '..png', 26:'..png'}}}
            # print(image2feat(bag[0]))
            bag_feat = {idx_image:image2feat(image) for idx_image, image in bag.items() if os.path.isfile(image2feat(image)) }
            bags_features[idx_bag] = bag_feat
            
        self.log.info(f"{self.class_name}.{'_get_bags_features'}: bags_features:{bags_features}")
        self.log.info(f"{self.class_name}.{'_get_bags_features'}: bags_features[0]:{bags_features[0]}")

        return bags_features

    
    def summary(self, bags_instances:dict, bags_labels:dict, bags_features:dict):

        ex_feat = np.load(bags_features[0][0])
        self.log.info(f"{self.class_name}.{'summary'}: Instance 0 of bag 0 has shape: {ex_feat.shape}")
        self.log.info(f"{self.class_name}.{'summary'}: Range:{(ex_feat.max(), ex_feat.min())}, mean: {ex_feat.mean()}")
        arr_labels = np.array(list(bags_labels.values()))
        self.log.info(f"{self.class_name}.{'summary'}: #Bags: {len(bags_instances)}, #pos_labels: {arr_labels.sum()}, #neg_labels: {len(bags_instances) - arr_labels.sum()}")
        # self.log.info(f"")
        return
    

    
    def __call__(self):

        bags_indices = self.get_bags_indices()
        bags_labels = self.get_fake_labels(bags_indices=bags_indices)
        bags_features = self._get_bags_features(bags_indices=bags_indices)

        # self.summary(bags_instances=bags_indices, bags_labels=bags_labels, bags_features=bags_features)
        # assert len(bags_indices) == len(bags_features) == len(bags_labels), f"Instances, features and labels don't have same length: instances: {len(bags_instances)}, features: {len(bags_features)}, labels: {len(bags_labels)}"
        # print(f"instances: {len(bags_instances)}, features: {len(bags_features)}, labels: {len(bags_labels)}")
        # print(f"instances: {type(bags_indices[0])}, features: {type(bags_features[0])}, labels: {type(bags_labels[0])}")


        return bags_indices, bags_features, bags_labels




def test_BagCreator():
    instances_folder = '/Users/marco/helical_tests/test_bagcreator/images'
    exp_folder = '/Users/marco/yolov5/runs/detect/exp7'
    sclerosed_idx=2
    n_images_per_bag = 9
    n_classes = 4
    creator = BagCreator(instances_folder=instances_folder, 
                         sclerosed_idx=sclerosed_idx, 
                         exp_folder=exp_folder,
                         n_classes=n_classes,
                         n_images_per_bag=n_images_per_bag)
    creator()

    return



if __name__ == '__main__':

    test_BagCreator()