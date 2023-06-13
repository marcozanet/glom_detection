import os
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
from MIL_bags_creation_new import BagCreator
from configurator import Configurator
from MIL_bag_manager import BagManager
from cnn_feat_extract_main import extract_cnn_features


class MILDataset(Dataset, Configurator):
    """ Dataset for bags of instances. 
        train_data = [bag_features, bag_label]
        -> given an idx it reads bag_features, bag_label"""


    def __init__(self,
                 folder:str,
                 all_slides_dir:str,
                 map_classes:dict, # {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2},
                 bag_classes:dict, #= {0:0.25, 1:0.5, 2:0.75, 3:1},
                 stain:str='pas',
                 n_instances_per_bag:int=9,
                 limit_n_bags_to:int = None ) -> None:
        
        super().__init__()
        self.folder = folder 
        self.all_slides_dir = all_slides_dir
        self.map_classes = map_classes
        self.bag_classes = bag_classes
        self.stain = stain.upper()
        self.n_instances_per_bag = n_instances_per_bag
        self.limit_n_bags_to = limit_n_bags_to

        self.data = self.get_data()

        return 
    
    
    def get_data(self):
        """ 1) 1st bag creation round 2) Balances bags by augmenting images 
            3) extract features for all resulting images 4) Converts bag instances from images to features. """

        # 1) create bags from images
        bag_manager = BagManager(folder=self.folder, 
                            map_classes=self.map_classes,
                            bag_classes=self.bag_classes,
                            all_slides_dir=self.all_slides_dir,
                            stain=self.stain, 
                            n_instances_per_bag=self.n_instances_per_bag)
        # bag_manager._del_augm_files()
        bag_manager()

        # 2) extract features from all images
        feat_extract_path_like = os.path.join(os.path.dirname(os.path.dirname(self.folder)), 'feat_extract')
        if os.path.isdir(feat_extract_path_like):
            print(f"Feat extract already existing: {feat_extract_path_like}")
        
        sets2extract = os.path.basename(self.folder) if os.path.basename(self.folder) in ['train', 'val', 'test'] else 'all'
        print(f"Extracting: {sets2extract}")
        extract_cnn_features(sets2extract=sets2extract)

        # 3) convert image bags to feature bags:
        bag_manager.convert_imagebags2featbags()
        bags_indices = bag_manager.bags_indices
        bags_labels = bag_manager.bags_labels

        print(f"bags labels idx0: {bags_labels[0]}")
        print(f"bag at idx 0: {bags_indices[0]}")

        data = []
        for i, (bag_idx, bag) in enumerate(bags_indices.items()):
            data.append((bag, bags_labels[bag_idx]))
            if self.limit_n_bags_to is not None:
                if i==self.limit_n_bags_to:
                    break
        # self.data = data
        # self._print_bag_summary()


        return  data    
    

    # def _print_bag_summary(self):
    #     """ Prints a summary for the created bags."""

    #     print(f"Created a total of {len(self.data)} bags. ")
    #     # bag_features, bag_label = self.__getitem__[0]
    #     n_feats = bag_features.shape[0]
    #     print(f"Each bag has {self.n_instances_per_bag} instances. Each instance has {n_feats//self.n_instances_per_bag} features, for a total {n_feats} features. ")

    #     return


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx) -> dict:

        # 1) read label:
        bag_label = self.data[idx][1]

        # 2) read feats and convert to torch:
        bag_features = []
        for file_idx, file in self.data[idx][0].items():
            file_feat = torch.from_numpy(np.load(file))
            bag_features.append(file_feat)
        bag_features = torch.stack(bag_features)
        bag_features = bag_features.flatten()

        return bag_features, bag_label


def test_MILDataset():

    folder = '/Users/marco/helical_tests/test_cnn_zaneta/cnn_dataset/test'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2}
    bag_classes = {0:0.25, 1:0.5, 2:0.75, 3:1}
    stain = 'PAS'
    n_instances_per_bag=9
    all_slides_dir='/Users/marco/Downloads/zaneta_files/safe'
    dataset = MILDataset(folder=folder, 
                        map_classes=map_classes,
                        bag_classes=bag_classes,
                        all_slides_dir=all_slides_dir,
                        stain=stain, 
                        n_instances_per_bag=n_instances_per_bag)

    print(len(dataset))

    return
    


if __name__ == '__main__':
    test_MILDataset()