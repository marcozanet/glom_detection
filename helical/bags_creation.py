import os 
from typing import Tuple, List
from glob import glob

import numpy as np
import random

class BagCreator(): 
    def __init__(self, 
                instances_folder:str, 
                exp_folder:str,
                sclerosed_idx: int,
                
                ) -> None:
        """ Given a root of instances and labels, 
        instances = folder with 2048x2048 tiles, labels = 0 if no SCLEROSED gloms within tile (i.e. tissue or healthy glom or glom NA).
        BagCreator will create 1 bag = 1 ROI of 4096x4096 -> the region is represented by the 9 tiles that fall within that region ."""
        
        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."       
       
        self.instances_folders = instances_folder
        self.sclerosed_idx = sclerosed_idx
        self.exp_folder = exp_folder

        return
    

    def get_bags_instances(self) -> dict: 

        images = glob(os.path.join(self.instances_folders, "*.png"))

        # get #samples per slide: 
        wsi_fnames = list(set([os.path.basename(file).split('_SFOG')[0] for file in images]))
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
                # print(f"files: {len(files)}")
                somma += len(files)

                # fill bags:
                n_bags = len(files) // 9 
                remaining = files
                for _ in range(n_bags):
                    selected = random.sample(remaining, k = 9)
                    remaining = [os.path.join(self.instances_folders, file) for file in remaining if file not in selected ]
                    bags[bag_n] = selected
                    bag_n += 1
                if len(remaining) > 0:
                    choosable = [file for file in files if file not in remaining]
                    remaining_bag = remaining + random.sample(choosable, k= 9-len(remaining) )
                    remaining_bag = [os.path.join(self.instances_folders, file) for file in remaining_bag ]
                    bags[bag_n] = remaining_bag
                
        # print(bags[0])
        return  bags
    

    def get_bags_labels(self, bags: dict):

        bags_labels = {}
        for idx, instances in bags.items():
            for instance in instances: 
                label_fp= instance.replace('images', 'labels')

                # if label doesn't exist, then no sclerosed glom.
                if not os.path.isfile(label_fp):
                    bags_labels[idx] = 0 
                    continue
                
                # open correspondent label:
                with open(label_fp, 'r') as f:
                    text = f.readlines()

                # bag label = 1 if at least one instance label = 1:
                label = 0
                for line in text:
                    clss = line.split(' ')[0]
                    if clss == self.sclerosed_idx:
                        label = 1 # i.e.
                bags_labels[idx] = label

        # print(bags_labels)
        
        return  bags_labels

    
    def _get_bag_features(self, bags_instances:dict): 
        """ Returns bag features. """

        bag_features = {}
        for bag, instances in bags_instances.items():
            samples = [file.replace('.png', '') for file in instances]
            print(samples)
            print((os.path.join(self.exp_folder, samples[0], "*.npy")))
            np_features = sorted([glob(os.path.join(self.exp_folder, sample, "*.npy")) for sample in samples])
            print(np_features)
            np_features = [feat_ls[-1] for feat_ls in np_features]
            bag_features[bag] = np_features

            raise NotImplementedError()

        print(bag_features)
        
        return

    
    def summary(self, bags_instances:dict, bags_labels:dict):

        arr_labels = np.array(list(bags_labels.values()))
        print(f"#Bags: {len(bags_instances)}, #pos_labels: {arr_labels.sum()}, #neg_labels: {len(bags_instances) - arr_labels.sum()}")

        return



    
    def __call__(self):

        bags_instances = self.get_bags_instances()
        bags_labels = self.get_bags_labels(bags = bags_instances)
        bags_labels = self._fake_labels_(bag_labels=bags_labels)
        bags_features = self._get_bag_features(bags_instances=bags_instances)

        self.summary(bags_instances=bags_instances, bags_labels=bags_labels)

        return bags_instances, bags_features, bags_labels


    def _fake_labels_(self, bag_labels:dict):

        print(f"Making some fake labels.")
        
        num_pos = 30
        pos_keys = random.sample(list(bag_labels.keys()), k = num_pos)
        for key in pos_keys: 
            bag_labels[key] = 1
        
        # print(bag_labels)

        return  bag_labels


def test_BagCreator():
    instances_folder = '/Users/marco/Downloads/try_train/detection/tiles/test/images'
    exp_folder = '/Users/marco/yolov5/runs/detect/exp2'
    sclerosed_idx=2
    creator = BagCreator(instances_folder=instances_folder, 
                        sclerosed_idx=sclerosed_idx, 
                        exp_folder=exp_folder)
    creator()

    return



if __name__ == '__main__':

    test_BagCreator()