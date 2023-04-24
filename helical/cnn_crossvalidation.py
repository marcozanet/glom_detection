import shutil
from abc import abstractmethod
from typing import Literal, List
import os
from configurator import Configurator
from crossvalidation import KCrossValidation
from glob import glob
import numpy as np
from tqdm import tqdm
import json


class CNN_KCrossValidation(KCrossValidation): 

    def __init__(self, 
                 *args, 
                 **kwargs
                 ) -> None:
        
        super().__init__(*args, **kwargs)
        self.class_name = self.__class__.__name__  
        # self.wsi_img_fmt = '.tif'
        # self.wsi_lbl_fmt = '.json'
        self.tile_img_fmt = '.jpg'
        # self.tile_lbl_fmt = '.txt'
        # self.data_root = data_root
        # self.k = k
        # self.dataset = dataset
        self._parse()

        # PRIMA C'ERA QUESTO:
        # self.data, self.file_list = self._get_data()
        # self.n_wsis = len(self.data['wsi_imgs'])
        # self.log.info(f"{self.class_name}.__init__: n_wsis:{self.n_wsis}.")
        # self.splits = self.create_folds()
        # self.n_tiles_dict = self.get_n_tiles()
        
        return
    
    def _parse(self): 

        assert self.k > 1, self.log.error(f"{self.class_name}.__init__: k:{self.k} should be at least 2.")
        assert os.path.isdir(self.data_root), self.log.error(f"{self.class_name}.__init__: data_root:{self.data_root} is not a valid dirpath.")

        return
    
    
    def _get_data(self): 

        tile_imgs = sorted(glob(os.path.join(self.data_root, 'tiles', '*', '*', f'*{self.tile_img_fmt}')))
        tile_lbls = [os.path.split(os.path.dirname(fp))[1] for fp in tile_imgs]
        wsi_fns = list(set([os.path.basename(fp).split('_')[0] for fp in tile_imgs]))

        assert len(wsi_fns) > 0, self.log.warning(f"{self.class_name}._get_data: len(wsi_imgs):{len(wsi_fns)}.")
        assert len(tile_imgs) > 0, self.log.warning(f"{self.class_name}._get_data: len(tile_imgs):{len(tile_imgs)}.")
        assert len(tile_lbls) > 0, self.log.warning(f"{self.class_name}._get_data: len(tile_lbls):{len(tile_lbls)}.")

        data = {'wsi_fns':wsi_fns, 'tile_imgs':tile_imgs, 'tile_lbls':tile_lbls}
        file_list = tile_imgs + tile_lbls

        return data, file_list
    

    def _get_folds(self) -> List[list]:
        """ Splits data into k folds. """

        list_imgs = sorted(self.data['wsi_fns'])
        fnames = sorted(self.data['wsi_fns'])#sorted([os.path.basename(name).split('.')[0] for name in list_imgs])
        # print(fnames)
        fp_fnames = {os.path.basename(fp).split('.')[0]:fp for fp in list_imgs}
        fp_fnames = dict(sorted(fp_fnames.items(), key=lambda x:x[0]))
        # print(fp_fnames)


        folds = []
        idxs = np.linspace(start=0, stop=self.n_wsis, num=self.k+1, dtype=int) # k + 1 because first idx=0
        folds_fnames = sorted([fnames[idxs[i]:idxs[i+1]] for i in range(self.k)])
        self.log.error(folds_fnames)
        folds = []
        for ls_fnames in folds_fnames:
            ls_fullnames = [fp_fnames[name] for name in sorted(ls_fnames)]
            folds.append(ls_fullnames)
        folds = sorted([list_imgs[idxs[i]:idxs[i+1]] for i in range(self.k)])

        assert len(folds) == self.k, self.log.err(f"{self.class_name}._get_folds: len(folds)={len(folds)} but should be =k={self.k}.")
        
        return folds
    
    def _get_i_split(self, folds:dict): 

        print(folds)

        fold_splits = {}
        for i in range(self.k): 
            test_idx = i
            train_idxs = sorted([j for j in range(self.k) if j!=i])
            print(train_idxs)
            train_folds = sorted([wsi for idx in train_idxs for wsi in sorted(folds)[idx] ])
            print(f"AHHHHH{train_folds}")
            # self.log.warning(train_folds)

            test_fold = folds[test_idx]
            fold_splits[i] = {'train': train_folds, 'val': test_fold}
            
            tot_images = len(fold_splits[i]['train']) + len(fold_splits[i]['val'])
            assert tot_images == self.n_wsis, self.log.error(f"{self.class_name}._get_i_split: sum of elems in train and test folder should be = self.n_wsis({tot_images}) but is .")
            # TODO add that train should be disjoint with test folds.
            self.log.info(f"{self.class_name}._get_i_split: iter_{i},  train: {len(fold_splits[i]['train'])} images, test: {len(fold_splits[i]['val'])}.")
            
        return fold_splits
    
    def _change_kfold(self, fold_i:int): 

        assert fold_i < self.k , self.log.error(f"{self.class_name}._change_folds: fold_i:{fold_i} should be < num folds({self.k}). Index starting from 0.")
        
        # ADDED HERE NOT TESTED, WAS IN THE INIT
        self.data, self.file_list = self._get_data() #self.file_list
        self.n_wsis = len(self.data['wsi_fns'])
        self.log.info(f"{self.class_name}.__init__: n_wsis:{self.n_wsis}.")
        self.splits = self.create_folds()
        self.n_tiles_dict = self.get_n_tiles()

        # for each wsi change folder to according new_folder 
        change_dir = lambda fp, set_fold: os.path.join(os.path.dirname(os.path.dirname((os.path.dirname(fp)))), set_fold, os.path.split(os.path.dirname(fp))[1], os.path.basename(fp))

        # self.log.info(f"{self.class_name}._change_folds: fold_i:{fold_i}. Moving files to new new folds.")
        train_basenames = [os.path.basename(name).split('.')[0] for name in self.splits[fold_i]['train']]
        test_basenames = [os.path.basename(name).split('.')[0] for name in self.splits[fold_i]['val']]
        self.log.info(f"{self.class_name}._change_folds: train: {len(train_basenames)} images: {train_basenames}.")
        self.log.info(f"{self.class_name}._change_folds: test: {len(test_basenames)} labels: {test_basenames}.")

        # TODO AGGIUNGI CHE LE FOLD DEVONO ESSERE DISJOINT TRA LORO 

        # raise NotImplementedError()

        def _move_files(to_fold:str) -> None:
            # basenames = train_basenames if to_fold == 'train' else test_basenames
            n_images = 0
            for wsi in tqdm(self.splits[fold_i][to_fold]): # get wsi of the files splitted in the folds
                train_old_new = [(fp, change_dir(fp, to_fold)) for fp in self.file_list if os.path.basename(wsi).split('.')[0] in fp] # take all files with same basename and changes whatever dir to 'train'.
                # train_old_new = [(fp, change_dir(fp, to_fold)) for fp in self.file_list if os.path.basename(wsi).split('.')[0] in fp] # take all files with same basename and changes whatever dir to 'train'.
                for old_fp, new_fp in train_old_new:
                    n_images +=1
                    if not os.path.isfile(new_fp):
                    # print(f"src:{old_fp}")
                    # print(f"dst:{new_fp}")
                        shutil.move(src=old_fp, dst=new_fp)
            return n_images
        
        n_train = _move_files('train')
        self.data, self.file_list = self._get_data() # update data
        self.log.info(f"Splitted train: {n_train}")

        # TODO ADD ONLY IN CASE THERE IS 
        n_train = _move_files('val')
        self.data, self.file_list = self._get_data() # update data
        self.log.info(f"Splitted val: {n_train}")

        return
    
    
    
    def create_folds(self): 
        """ Create and splits folds. """

        folds = self._get_folds()
        print(folds)
        fold_splits = self._get_i_split(folds=folds)

        return fold_splits


    

def test_KCrossValidation():

    data_root = '/Users/marco/helical_tests/test_cnn_processor/test_crossvalidation'
    k=3
    kcross = CNN_KCrossValidation(data_root=data_root, k=k)
    kcross._change_kfold(1)

    return

if __name__ == '__main__': 
    test_KCrossValidation()

        