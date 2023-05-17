import shutil
from abc import abstractmethod
from typing import Literal, List
import os
from configurator import Configurator
from glob import glob
import numpy as np
from tqdm import tqdm
import json


class KCrossValidation(Configurator): 

    def __init__(self, 
                 data_root:str,
                 dataset: Literal['hubmap', 'muw'], 
                 k:int = 5,
                 task = Literal['segmentation', 'detection'],
                 ) -> None:
        
        super().__init__()
        self.class_name = self.__class__.__name__  
        self.wsi_img_fmt = '.tif'
        self.wsi_lbl_fmt = '.json'
        self.tile_img_fmt = '.png'
        self.tile_lbl_fmt = '.txt'
        self.data_root = data_root
        self.k = k
        self.dataset = dataset
        self._parse()

        # self.data, self.file_list = self._get_data()
        # # self.log.info(len(self.file_list))
        # self.n_wsis = len(self.data['wsi_imgs'])
        # self.log.info(f"{self.class_name}.__init__: n_wsis:{self.n_wsis}.")
        # self.splits = self.create_folds()
        # self.n_tiles_dict = self.get_n_tiles()
        
        #self.log.info(f"{self.class_name}.: :{}.")
        return
    
    def _parse(self): 

        assert self.k > 1, self.log.error(f"{self.class_name}.__init__: k:{self.k} should be at least 2.")
        assert os.path.isdir(self.data_root), self.log.error(f"{self.class_name}.__init__: data_root:{self.data_root} is not a valid dirpath.")

        return
    

    
    def _get_data(self): 

        if self.dataset == 'hubmap':
            wsi_imgs = sorted(glob(os.path.join(self.data_root, 'wsi', '*', 'images', f'*{self.wsi_img_fmt}')))
            wsi_lbls = sorted(glob(os.path.join(self.data_root, 'wsi', '*', 'labels', '*')))
            tile_imgs = sorted(glob(os.path.join(self.data_root, 'tiles', '*', 'images', f'*{self.tile_img_fmt}')))
            tile_lbls = sorted(glob(os.path.join(self.data_root, 'tiles', '*', 'labels', f'*{self.tile_lbl_fmt}')))
        elif self.dataset == 'muw':
            wsi_imgs = sorted(glob(os.path.join(self.data_root, 'wsi', '*', '*', f'*{self.wsi_img_fmt}')))
            wsi_lbls = sorted(glob(os.path.join(self.data_root, 'wsi', '*', 'labels', '*_sample[0-8].json')))
            tile_imgs = sorted(glob(os.path.join(self.data_root, 'tiles', '*', 'images', f'*{self.tile_img_fmt}')))
            tile_lbls = sorted(glob(os.path.join(self.data_root, 'tiles', '*', 'labels', f'*{self.tile_lbl_fmt}')))
            other_wsi_lbls = sorted(glob(os.path.join(self.data_root, 'wsi', '*', 'labels', '*_sample[0-8]*')))

        assert len(wsi_imgs) > 0, f"{self.class_name}._get_data: 'wsi_imgs' is empty. No 'wsi_img' like {os.path.join(self.data_root, 'wsi', '*', 'images', f'*{self.wsi_img_fmt}')} ."
        assert len(wsi_imgs) > 0, f"{self.class_name}._get_data: 'wsi_lbls' is empty. No 'wsi_lbl' like {os.path.join(self.data_root, 'wsi', '*', 'labels', '*')} ."
        assert len(tile_imgs) > 0, f"{self.class_name}._get_data: 'tile_imgs' is empty. No 'tile_img' like {os.path.join(self.data_root, 'tiles', '*', 'images', f'*{self.tile_img_fmt}')} ."
        assert len(tile_lbls) > 0, f"{self.class_name}._get_data: 'tile_lbls' is empty. No 'tile_lbl' like {os.path.join(self.data_root, 'tiles', '*', 'labels', f'*{self.tile_lbl_fmt}')} ."
        if self.dataset == 'muw':
            assert len(other_wsi_lbls) > 0, f"{self.class_name}._get_data: 'other_wsi_lbls' is empty. No 'other_wsi_lbl' like {os.path.join(self.data_root, 'wsi', '*', 'labels', '*_sample[0-8]*')} ."

        data = {'wsi_imgs':wsi_imgs, 'wsi_lbls':wsi_lbls, 'tile_imgs':tile_imgs, 'tile_lbls':tile_lbls}
        file_list = wsi_imgs + wsi_lbls + tile_imgs + tile_lbls if self.dataset == 'hubmap' else  wsi_imgs + wsi_lbls + tile_imgs + tile_lbls + other_wsi_lbls

        return data, file_list
    

    def _get_folds(self) -> List[list]:
        """ Splits data into k folds. """

        if self.dataset == 'hubmap':
            list_imgs = sorted(self.data['wsi_imgs'])
        elif self.dataset == 'muw':
            list_imgs = sorted(self.data['wsi_lbls']) # tif are wsi, while labels are _samples
            # print(list_imgs)

        fnames = sorted([os.path.basename(name).split('.')[0] for name in list_imgs])
        fp_fnames = {os.path.basename(fp).split('.')[0]:fp for fp in list_imgs}
        # print(fnames)
        # print(len(fnames))
        # raise NotImplementedError()
        folds = []
        idxs = np.linspace(start=0, stop=self.n_wsis, num=self.k+1, dtype=int) # k + 1 because first idx=0
        # print(idxs)
        folds_fnames = [fnames[idxs[i]:idxs[i+1]] for i in range(self.k)]

        print(folds_fnames)
        folds = []
        for ls_fnames in folds_fnames:
            ls_fullnames = [fp_fnames[name] for name in ls_fnames]
            folds.append(ls_fullnames)
        # folds_fullnames = [fp for fp in ]
        # folds = [list_imgs[idxs[i]:idxs[i+1]] for i in range(self.k)]
        folds = [list_imgs[idxs[i]:idxs[i+1]] for i in range(self.k)]
        # print(f"folds: {len(folds)}")

        assert len(folds) == self.k, self.log.err(f"{self.class_name}._get_folds: len(folds)={len(folds)} but should be =k={self.k}.")

        # for fold in folds: 
        #     self.log.info(f"length fold: {len(fold)}")

        # print(folds)
        # raise NotImplementedError()
        # print(folds[0])
        return folds
    
    def _get_i_split(self, folds:dict): 

        # fold_splits = {i: {'train': folds[[j for j in range(self.k) if j != i]], 'val': folds[i] } for i in range(self.k) }    

        fold_splits = {}
        for i in range(self.k): 
            test_idx = i
            train_idxs = [j for j in range(self.k) if j!=i]
            # print(f'test:{test_idx}')
            # print(f"train:{train_idxs}")
            train_folds = [wsi for idx in train_idxs for wsi in folds[idx] ]

            # train_folds = [train_folds.append(folds[idx]) for idx in train_idxs]
            # print(len(train_folds))
            test_fold = folds[test_idx]
            # print(len(test_fold))
            fold_splits[i] = {'train': train_folds, 'val': test_fold}
            
            tot_images = len(fold_splits[i]['train']) + len(fold_splits[i]['val'])
            # assert tot_images == self.n_wsis, self.log.error(f"{self.class_name}._get_i_split: sum of elems in train and test folder should be = {self.n_wsis}, but is {tot_images}).")
            # TODO add that train should be disjoint with test folds.
            self.log.info(f"{self.class_name}._get_i_split: iter_{i},  train: {len(fold_splits[i]['train'])} images, test: {len(fold_splits[i]['val'])}.")
            
            # print(f"len fold: train:{len(fold_splits[i]['train'])}, test:{len(fold_splits[i]['val'])}")

        return fold_splits
    
    def _change_kfold(self, fold_i:int): 

        assert fold_i < self.k , self.log.error(f"{self.class_name}._change_folds: fold_i:{fold_i} should be < num folds({self.k}). Index starting from 0.")
        
        self.data, self.file_list = self._get_data()
        self.n_wsis = len(self.data['wsi_imgs'])
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

        def _move_files(to_fold:str) -> None:
            basenames = train_basenames if to_fold == 'train' else test_basenames
            for wsi in tqdm(self.splits[fold_i][to_fold]): # get wsi of the files splitted in the folds
                train_old_new = [(fp, change_dir(fp, to_fold)) for fp in self.file_list if os.path.basename(wsi).split('.')[0] in fp] # take all files with same basename and changes whatever dir to 'train'.
                # print(train_old_new[0])
                # raise NotImplementedError()
                for old_fp, new_fp in train_old_new:
                    if not os.path.isfile(new_fp):
                        shutil.move(src=old_fp, dst=new_fp)
            return
        
        _move_files('train')
        self.data, self.file_list = self._get_data() # update data
        self.log.info('train')
        # TODO ADD ONLY IN CASE THERE IS 
        self.create_n_tiles_file(train_basenames, fold='train')
        _move_files('val')
        self.log.info('val')
        self.create_n_tiles_file(test_basenames, fold='val')
        self.data, self.file_list = self._get_data() # update data

        # raise NotImplementedError()

        return
    
    
    def create_n_tiles_file(self, wsi_basenames:list, fold:str) -> None: 
        
        if self.dataset == 'muw': 
            wsi_basenames = [f"{basename}_sample{j}" for basename in wsi_basenames for j in range(5)]
            wsi_basenames = glob(os.path.join(self.data_root, 'wsi', fold, 'labels', '*_sample[0-9].json'))
            assert len(wsi_basenames)>0, f"'wsi_basenames' is empty. No files like {os.path.join(self.data_root, 'wsi', fold, 'labels', '*_sample[0-9].json')} "
            wsi_basenames = [os.path.basename(fp).split('.')[0] for fp in wsi_basenames]
            print(wsi_basenames)
        write_dict = {wsi_name:self.n_tiles_dict[wsi_name] for wsi_name in wsi_basenames}
        self.log.info(f"{write_dict}")
        fp = os.path.join(self.data_root, 'wsi', fold, 'labels', 'n_tiles.json')
        self.log.info(fp)

        if not os.path.isfile(fp):
            return


        json_obj = json.dumps(write_dict)

        with open(fp, 'w') as f:
            f.write(json_obj)

        return
    
    
    def get_n_tiles(self) -> dict: 

        ntiles_files = glob(os.path.join(self.data_root, 'wsi', '*', 'labels', 'n_tiles.json'))
        n_tiles_tot = {}
        for file in ntiles_files:  
            with open(file, 'r') as f:
                dictionary = json.load(f)
            n_tiles_tot.update(dictionary)

        self.log.info(f"{self.class_name}.fix_n_tiles: n_tiles_tot:{n_tiles_tot}.")

        return n_tiles_tot
    
    
    def create_folds(self): 
        """ Create and splits folds. """

        folds = self._get_folds()
        fold_splits = self._get_i_split(folds=folds)

        return fold_splits
    


    

def test_KCrossValidation():

    data_root = '/Users/marco/helical_tests/test_kcrossvalidation/detection'
    k=3
    kcross = KCrossValidation(data_root=data_root, k=k)
    kcross._change_kfold(1)
    # kcross._change_folds(1)
    # kcross._change_folds(0)
    # kcross._change_folds(2)
    # kcross._change_folds(0)

    return

if __name__ == '__main__': 
    test_KCrossValidation()

        