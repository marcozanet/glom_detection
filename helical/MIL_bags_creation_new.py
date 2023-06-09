import os 
from typing import Tuple, List
from glob import glob
import numpy as np
import random
from configurator import Configurator
import math
from tqdm import tqdm
import json
import math 

class BagCreator(Configurator): 
    
    def __init__(self, 
                instances_folder: str, 
                # sclerosed_idx: int,
                map_classes: dict,
                n_instances_per_bag: int,
                # n_classes: int,
                all_slides_dir:str,
                instance_fmt: str = '.npy',
                img_fmt: str = '.jpg',
                stain:str = 'pas',
                bag_classes:dict = {0:0.25, 1:0.5, 2:0.75, 3:1}, 
                ) -> None:
        
        """ Given a root of instances and labels, 
        instances = folder with 2048x2048 tiles, labels = 0 if no SCLEROSED gloms within tile (i.e. tissue or healthy glom or glom NA).
        BagCreator will create 1 bag = 1 ROI of 4096x4096 -> the region is represented by the 9 tiles that fall within that region ."""
        
        super().__init__()

        assert os.path.isdir(instances_folder), f"'instances_folders':{instances_folder} is not a valid dirpath."
        # assert isinstance(sclerosed_idx, int), f"'sclerosed_idx':{sclerosed_idx} should be an int."
        # assert os.path.isdir(exp_folder), f"'exp_folder':{exp_folder} is not a valid dirpath."       
       
        self.instances_folders = instances_folder
        # self.sclerosed_idx = sclerosed_idx
        # self.exp_folder = exp_folder
        self.class_name = self.__class__.__name__
        self.n_instances_per_bag = n_instances_per_bag
        # self.n_classes = n_classes
        self.img_fmt = img_fmt
        self.instance_fmt = instance_fmt
        self.map_classes = map_classes
        self.inversed_map_classes = {v:k for k,v in map_classes.items()}
        self.stain = stain.upper()
        self.bag_classes = bag_classes
        self.instance_path_like = os.path.join(self.instances_folders, '*',  'feats', f'*{self.instance_fmt}') if self.instance_fmt=='.npy' else os.path.join(self.instances_folders, '*',  f'*{self.instance_fmt}')

        self.wsi_labels = self._get_real_labels(all_slides_dir)
        self.instances_idcs = self.get_instances_indices()
        # self._get_file_labels()

        return
    
    
    def get_instances_indices(self) -> dict: 
        """ Compute each bag instances like {/User/.../instance0:0, /User/.../instance6:1... } """

        # print(self.instances_folders)
        # instance_path_like = os.path.join(self.instances_folders, '*',  'feats', f'*{self.instance_fmt}') if self.instance_fmt=='.npy' else os.path.join(self.instances_folders, '*',  f'*{self.instance_fmt}')
        instances = glob(self.instance_path_like)
        self.all_images = instances
        assert (len(instances)>0), f"0 instances like {self.instance_path_like} "
        # self.log.info(f"{self.class_name}.{'get_instances_indices'}: Total instances: {len(instances)}. Instances per bag: {self.n_instances_per_bag}, n_bags: {math.ceil(len(instances)/self.n_instances_per_bag)}. Repeating {len(instances)%self.n_instances_per_bag} to fill last folder")
        self.n_bags = math.ceil(len(instances)/self.n_instances_per_bag)
        assert self.n_bags>0, f"'n_bags':{self.n_bags} should be >0."
        instances_idcs = {fp:i for (i,fp) in enumerate(instances) }

        # print(f"instances_idcs:{instances_idcs} ")

        return instances_idcs



    def get_bags_indices(self) -> dict: 
        
        # random.seed(10)
        # instance_path_like = os.path.join(self.instances_folders, '*',  'feats', f'*{self.instance_fmt}') if self.instance_fmt == '.npy' else os.path.join(self.instances_folders, '*',   f'*{self.instance_fmt}')
        instances = glob(self.instance_path_like)
        assert os.path.split(os.path.dirname(self.instance_path_like)) or (len(instances)>0), f"0 instances like {self.instance_path_like} "

        # get #samples per slide:
        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]
        wsi_fnames = list(set([get_wsi_fname(file) for file in instances]))
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
        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]
        wsi_fnames = list(set([get_wsi_fname(file) for file in instances]))
        somma = 0
        bags = {}
        bag_n = 0
        for wsi, n_samples in tqdm(wsi_samples.items(), desc='Creating bags'):
            for i in range(n_samples):
                if self.multisample:
                    print(f'Multisample')
                    files = sorted([ os.path.basename(file) for file in instances if f"{wsi}_SFOG_sample{i}" in file])
                else:
                    files = [file for file in instances if wsi in file]
                assert len(files)>0, f"'files': {files} is empty." 
                somma += len(files)

                # Fill bags:
                n_bags = len(files) // self.n_instances_per_bag
                assert n_bags!=0, f"Files in {wsi} are insufficient to create a bag. At least 9 instances are required. Skipping slide."
                remaining = files  #[os.path.join(self.instances_folders, get_fname_class(file)[0], get_fname_class(file)[1], 'feats', file ) for file in instances]
                assert all([os.path.isfile(file) for file in remaining]), f"Not all files in 'remaining' are valid filepaths. E.g. of file: {remaining[0]}"
                for _ in range(n_bags):
                    selected = random.sample(remaining, k = self.n_instances_per_bag)
                    remaining = [file for file in remaining if file not in selected ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in selected}
                    bag_n += 1
                # the last bag won't be fully filled, so we'll re-sample some images to fill it:
                if len(remaining) > 0: # last bag
                    choosable = [file for file in files if file not in remaining]
                    remaining_bag = remaining + random.sample(choosable, k= self.n_instances_per_bag-len(remaining) )
                    remaining_bag = [file for file in remaining_bag ]
                    bags[bag_n] = {self.instances_idcs[file]: file for file in remaining_bag} # fill last bag
        
        
        assert len(bags) > 0, f"'bags':{bags} are empty."
        # self.log.info(f"Bags indices completed: created {len(bags)} bags. E.g.: {[(k,v) for k, v in bags.items() if k == 0]}")
        print(f"Created {len(bags)} bags.")
        return  bags
    
    # def _get_file_labels(self): 

    #     get_file_label = lambda fp: {fp:self.map_classes[os.path.split(os.path.dirname(os.path.dirname(fp)))[1]]}
    #     file_labels = {}
    #     for img in self.all_images:
    #         file_labels.update(get_file_label(img))
    #     # print(file_labels)

    #     self.file_labels = file_labels

    #     return file_labels

    def _del_empty_wsi(self, slide_name:str):
        """ Deletes crops coming from empty WSIs (don't have a label to be used.)"""

        instances = glob(self.instance_path_like)
        assert len(instances)>0, f"0 instances like: {self.instance_path_like}"
        del_images = [file for file in instances if slide_name in file ]
        for file in (del_images):
            os.remove(file)
            print(f"Removed files containing {slide_name}")

        return
    
    def _get_real_labels(self, all_slides_dir:str):
        """ Gets label for each WSI. For the dummy training, the 'real' labels are the stratification 
            of the number of unhealthy gloms, i.e. similarly to Berden score."""


        annotations = glob(os.path.join(all_slides_dir, '*.json'))
        # print(self.map_classes)
        possible_classes = {k:v for k,v in self.map_classes.items() if k!='false_positives'}
        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('.json')[0] + '_'

        stratify = lambda ratio: math.floor(ratio*4)

        slides_labels = {}
        # slides_to_skip = []
        for annotation in tqdm(annotations, desc='Getting real labels'): 

            slide_name = get_wsi_fname(annotation)
            # print(get_wsi_fname(annotation))
            with open(annotation, 'r') as f:
                json_obj = json.load(f)
            
            slide_label = {k:0 for k,_ in self.map_classes.items() if k!='false_positives'}
            for obj_dict in json_obj:
                clss = obj_dict['properties']['classification']['name']
                assert clss in possible_classes, f"class is {clss} but possible classes are: {possible_classes}"
                slide_label[clss]+=1
            
            # print(slide_label)
            if slide_label['Glo-unhealthy'] + slide_label['Glo-healthy'] == 0:
                # slides_to_skip.append(slide_name)
                # print(f"Can't retrieve for slide {slide_name}. Deleting file. ")
                self._del_empty_wsi(slide_name)
                continue

            slide_label = stratify(slide_label['Glo-unhealthy']/(slide_label['Glo-healthy'] + slide_label['Glo-unhealthy']) )
            slides_labels.update({slide_name:slide_label})

        self.slides_labels = slides_labels

        return

    
    def get_bags_labels(self): 

        get_wsi_fname = lambda fp: os.path.basename(fp).split(self.stain)[0].split('Augm')[1][1:].split('_',1)[1] if 'Augm' in fp else os.path.basename(fp).split(self.stain)[0]
        
        bags_labels = {}
        for bag_idx, bag in tqdm(self.bags_indices.items(), desc='Assigning labels to bags'):
            slide_name = get_wsi_fname(list(bag.values())[0])
            assert all([slide_name == get_wsi_fname(val) for val in bag.values()])
            bag_label = {bag_idx:self.slides_labels[slide_name]}
            bags_labels.update(bag_label)


        self.bags_labels = bags_labels

        return 
    
    
    # def _get_bags_features(self, bags_indices:dict): 
    #     """ Returns bag features. """


    #     # create features folds:
    #     image2feat = lambda img_file: os.path.join(os.path.dirname(img_file), os.path.basename(img_file).replace(self.img_fmt, '.npy')) 
    #     bags_features = {}
    #     for idx_bag, bag in tqdm(bags_indices.items(), desc='creating features'):
    #         bag_feat = {idx_image:image2feat(image) for idx_image, image in bag.items()  }
    #         bag_feat = {k:v for k,v in bag_feat.items() if os.path.isfile(v)}
    #         assert len(bag_feat)>0, f"'bag_feat' has length 0, but should be > 0. "
    #         bags_features[idx_bag] = bag_feat
            
    #     assert len(bags_features)>0, f"'bags_features' has length 0, but should be > 0. "
    #     self.log.info(f"Created features.")

    #     self.bags_features = bags_features

    #     return bags_features

    
    def summary(self, bags_instances:dict, bags_labels:dict, bags_features:dict):

        ex_feat = np.load(bags_features[0][0])
        self.log.info(f"{self.class_name}.{'summary'}: Instance 0 of bag 0 has shape: {ex_feat.shape}")
        self.log.info(f"{self.class_name}.{'summary'}: Range:{(ex_feat.max(), ex_feat.min())}, mean: {ex_feat.mean()}")
        arr_labels = np.array(list(bags_labels.values()))
        self.log.info(f"{self.class_name}.{'summary'}: #Bags: {len(bags_instances)}, #pos_labels: {arr_labels.sum()}, #neg_labels: {len(bags_instances) - arr_labels.sum()}")
        return
    

    
    def __call__(self):
        """ Creates bags (bags indices) and bags labels. Each bag is associated with the 
            label of the WSI/ROI that the images in the bag belong to."""
        

        # 1) create bags 
        self.bags_indices = self.get_bags_indices()
        # bags_features = self._get_bags_features(bags_indices=bags_indices)
        # self.bags_labels = self.get_fake_labels(bags_indices=self.bags_indices)
        # self.bags_labels = self._get_real_labels(all_slides_dir='/Users/marco/Downloads/zaneta_files/safe')
        
        # 2) get bags labels 
        self.get_bags_labels()

        assert len(self.bags_indices)>0, f"'bags_indices': {self.bags_indices} is empty." 
        assert len(self.bags_labels)>0, f"'bags_labels': {self.bags_labels} is empty." 
        # assert len(bags_features)>0, f"'bags_features': {bags_features} is empty." 
        assert len(self.bags_indices)==len(self.bags_labels), f"Lengths of 'bags_indices':{len(self.bags_indices)}, 'bags_labels':{len(bags_labels)}, should be the same. "
        
        return self.bags_indices, self.bags_labels




def test_BagCreator():
    instances_folder = '/Users/marco/helical_tests/test_cnn_zaneta/cnn_dataset'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2}
    # sclerosed_idx=2
    n_instances_per_bag = 9
    # n_classes = 4
    all_slides_dir='/Users/marco/Downloads/zaneta_files/safe'
    creator = BagCreator(instances_folder=instances_folder, 
                        #  sclerosed_idx=sclerosed_idx, 
                         map_classes=map_classes,
                        #  n_classes=n_classes,
                         n_instances_per_bag=n_instances_per_bag,
                         all_slides_dir=all_slides_dir,
                         instance_fmt='.jpg',
                         img_fmt='.jpg')

    creator()
    print(creator.bags_labels)

    return



if __name__ == '__main__':

    test_BagCreator()