import os 
from typing import Tuple, List
from glob import glob
import numpy as np
import random
from configurator import Configurator
import math
from tqdm import tqdm

class BagCreator(Configurator): 
    
    def __init__(self, 
                instances_folder:str, 
                sclerosed_idx: int,
                map_classes: dict,
                n_instances_per_bag: int,
                n_classes: int,
                instance_fmt: str = '.npy',
                img_fmt: str = '.jpg',
                ) -> None:
        
        """ Given a root of instances and labels, 
        instances = folder with 2048x2048 tiles, labels = 0 if no SCLEROSED gloms within tile (i.e. tissue or healthy glom or glom NA).
        BagCreator will create 1 bag = 1 ROI of 4096x4096 -> the region is represented by the 9 tiles that fall within that region ."""
        
        super().__init__()

        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        # assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."       
       
        self.instances_folders = instances_folder
        self.sclerosed_idx = sclerosed_idx
        # self.exp_folder = exp_folder
        self.class_name = self.__class__.__name__
        self.n_instances_per_bag = n_instances_per_bag
        self.n_classes = n_classes
        self.img_fmt = img_fmt
        self.instance_fmt = instance_fmt
        self.map_classes = map_classes

        self.instances_idcs = self.get_instances_indices()

        return
    
    
    def get_instances_indices(self) -> dict: 
        """ Compute each bag instances like {/User/.../instance0:0, /User/.../instance6:1... } """

        # print(self.instances_folders)
        instance_path_like = os.path.join(self.instances_folders, '*', 'feats', f'*{self.instance_fmt}')
        instances = glob(instance_path_like)
        assert os.path.split(os.path.dirname(instance_path_like)) or (len(instances)>0), f"0 instances like {instance_path_like} "
        self.log.info(f"{self.class_name}.{'get_instances_indices'}: Total instances: {len(instances)}. Instances per bag: {self.n_instances_per_bag}, n_bags: {math.ceil(len(instances)/self.n_instances_per_bag)}. Repeating {len(instances)%self.n_instances_per_bag} to fill last folder")
        self.n_bags = math.ceil(len(instances)/self.n_instances_per_bag)
        assert self.n_bags>0, f"'n_bags':{self.n_bags} should be >0."
        instances_idcs = {fp:i for (i,fp) in enumerate(instances) }

        print(f"instances_idcs:{instances_idcs} ")

        return instances_idcs



    def get_bags_indices(self) -> dict: 

        instance_path_like = os.path.join(self.instances_folders, '*', 'feats', f'*{self.instance_fmt}')
        instances = glob(instance_path_like)
        assert os.path.split(os.path.dirname(instance_path_like)) or (len(instances)>0), f"0 instances like {instance_path_like} "

        # get #samples per slide: 
        wsi_fnames = list(set([os.path.basename(file).split('_')[0] for file in instances]))
        print(wsi_fnames)
        assert len(wsi_fnames)>0, f"'wsi_fnames': {wsi_fnames} is empty." 
        assert all([self.instance_fmt not in file for file in wsi_fnames]), f"'wsi_fnames':{wsi_fnames} shouldn't contain {self.instance_fmt}"
        
        wsi_samples = {}
        for wsi in wsi_fnames:
            if 'sample' in wsi:
                samples = [int(file.split('sample')[1].split('_')[0]) for file in instances if wsi in file]
                wsi_samples[wsi] = np.array(samples).max() + 1
                assert len(wsi_samples)>0, f"'wsi_samples': {wsi_samples} is empty." 
                self.multisample = True
            else:
                wsi_samples[wsi] = 1
                self.multisample = False

        # Create bags:
        def get_fname_class(fn): # helper func
            get_class = lambda fp: os.path.split(os.path.dirname(os.path.dirname(fp)))[1]
            assert get_class(instances[0]) in list(self.map_classes.keys()), f"'{get_class(instances[0])}' not in {list(self.map_classes.keys())}."
            matches = [get_class(fp) for fp in instances if fn in fp]
            assert len(matches)>0, f"No fp matches for the fn:{fn}. "
            assert len(matches)==1, f"More than 1 fp match for fn:{fn}"
            return matches[0]
        
        somma = 0
        bags = {}
        bag_n = 0
        for wsi, n_samples in tqdm(wsi_samples.items(), desc='Creating bags'):
            for i in range(n_samples):
                if self.multisample:
                    files = sorted([ os.path.basename(file) for file in instances if f"{wsi}_SFOG_sample{i}" in file])
                else:
                    files = sorted([ os.path.basename(file) for file in instances if f"{wsi}" in file])

                # print(files[0])
                assert len(files)>0, f"'files': {files} is empty." 
                somma += len(files)
                print(somma)

                # Fill bags:
                n_bags = len(files) // self.n_instances_per_bag
                remaining = [os.path.join(self.instances_folders, get_fname_class(file), 'feats', file ) for file in files]
                assert all([os.path.isfile(file) for file in remaining]), f"Not all files in 'remaining' are valid filepaths. E.g. of file: {remaining[0]}"
                for _ in range(n_bags):
                    selected = random.sample(remaining, k = self.n_instances_per_bag)
                    remaining = [file for file in remaining if file not in selected ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in selected}
                    # bags[bag_n] = [(self.instances_idcs[file],file) for file in selected]
                    bag_n += 1
                # the last bag won't be fully filled, so we'll re-sample some images to fill it:
                if len(remaining) > 0: # last bag
                    choosable = [file for file in files if file not in remaining]
                    remaining_bag = remaining + random.sample(choosable, k= self.n_instances_per_bag-len(remaining) )
                    remaining_bag = [os.path.join(self.instances_folders, get_fname_class(file), file) for file in remaining_bag ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in remaining_bag} # fill last bag
        
        
        assert len(bags) > 0, f"'bags':{bags} are empty."
        print(f"bags:{bags}")

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
        # self.log.info(f"{self.class_name}.{'get_fake_labels'}: n_classes:{self.n_classes}")
        self.log.info(f"{self.class_name}.{'get_fake_labels'}: bag_labels:{bags_labels}")
        assert len(bags_labels)>0, f"'bags_labels' has length 0, but should be > 0. "
        print(f"bags_labels: {bags_labels}")

        return bags_labels

    
    # def _get_bags_features(self, bags_indices:dict): 
    #     """ Returns bag features. """


    #     # create features folds:
    #     for _fold in ['train', 'val', 'test']:
    #         os.makedirs(os.path.join(os.path.dirname(self.exp_folder), _fold, 'feats'), exist_ok=True)

        # image2feat = lambda img_file: os.path.join(self.exp_folder, 'feats', os.path.basename(img_file).replace(self.img_fmt, '.npy'))
        # bags_features = {}
        # for idx_bag, bag in tqdm(bags_indices.items(), desc='creating features'):
        #     # print(idx_bag)
        #     # print(bag)
        #     # print(image2feat('/Users/marco/helical_tests/test_bagcreator/images/200701099_09_SFOG_sample0_0_4.png'))
        #     # bags_idcs like {0: {0: '....png', 6: '..png', 16:'..png'}, 1:{19: '....png', 1: '..png', 26:'..png'}}}
        #     # print(image2feat(bag[0]))
        #     # bag_feat = [(image2feat(image)) for idx_image, image in bag.items()  ]
        #     # print(bag_feat)
        #     bag_feat = {idx_image:image2feat(image) for idx_image, image in bag.items()  }
        #     # print(bag_feat)
        #     assert len(bag_feat)>0, f"'bag_feat' has length 0, but should be > 0. "
        #     bags_features[idx_bag] = bag_feat
            
        # # self.log.info(f"{self.class_name}.{'_get_bags_features'}: bags_features:{bags_features}")
        # # self.log.info(f"{self.class_name}.{'_get_bags_features'}: bags_features[0]:{bags_features[0]}")
        # assert len(bags_features)>0, f"'bags_features' has length 0, but should be > 0. "
        # print(f"bags_features: {bags_features}")
        # # print('bags features done')

        # return bags_features

    
    def summary(self, bags_instances:dict, bags_labels:dict, bags_features:dict):

        ex_feat = np.load(bags_features[0][0])
        self.log.info(f"{self.class_name}.{'summary'}: Instance 0 of bag 0 has shape: {ex_feat.shape}")
        self.log.info(f"{self.class_name}.{'summary'}: Range:{(ex_feat.max(), ex_feat.min())}, mean: {ex_feat.mean()}")
        arr_labels = np.array(list(bags_labels.values()))
        self.log.info(f"{self.class_name}.{'summary'}: #Bags: {len(bags_instances)}, #pos_labels: {arr_labels.sum()}, #neg_labels: {len(bags_instances) - arr_labels.sum()}")
        # self.log.info(f"")
        return
    

    
    def __call__(self):
        
        if os.path.split(self.instances_folders)[1] == 'test':
            return

        bags_indices = self.get_bags_indices()
        bags_labels = self.get_fake_labels(bags_indices=bags_indices)
        bags_features = self._get_bags_features(bags_indices=bags_indices)

        assert len(bags_indices)>0, f"'bags_indices': {bags_indices} is empty." 
        assert len(bags_labels)>0, f"'bags_labels': {bags_labels} is empty." 
        assert len(bags_features)>0, f"'bags_features': {bags_features} is empty." 
        assert len(bags_indices)==len(bags_labels)==len(bags_features), f"Lengths of 'bags_indices':{len(bags_indices)}, 'bags_labels':{len(bags_labels)}, 'bags_features':{len(bags_features)} should be the same. "
        
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