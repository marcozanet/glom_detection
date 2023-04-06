import os 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from profiler_base import ProfilerBase
import json

class ProfilerHubmap(ProfilerBase): 

    def __init__(self, 
                *args,
                **kwargs) -> None:
        """ Data Profiler to help visualize a data overview. 
            Needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels.
            '"""
        
        other_params = {'wsi_images_like':'*.tif', 'wsi_labels_like': '*.txt', 
                        'tile_images_like':'*.png', 'tile_labels_like':'*.txt'}
        kwargs.update(other_params)
        super().__init__(*args, **kwargs)
        self.n_classes = 1

        return

    
    def _get_instances_df(self) -> pd.DataFrame: 
        """ Creates the tiles DataFrame. """
        col_names = ['class_n','width','height','box_area', 'obj_n', 'tile', 'fold', 'wsi', 'fn' ]
        df = pd.DataFrame(columns=col_names)

        # open all labels: 
        i = 0
        label_f = self.data['tile_labels']
        for file in tqdm(label_f, desc = 'Creating df_instances'):

            # read label:
            with open(file, 'r') as f:
                rows = f.readlines()
            assert len(rows) <= 30, self.log.warning(f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? ")

            # get info from label:
            fn = os.path.basename(file)
            rows = [row.replace('\n', '') for row in rows]
            for row in rows:
                items = row.split(' ')
                class_n = items[0]
                items = items[1:]
                assert len(items)%2 == 0, self.log.error(f"{self._class_name}.{'_get_instances_df'}: items in row are not even.")
                x = [float(el) for (j,el) in enumerate(items) if j%2 == 0]
                y = [float(el) for (j,el) in enumerate(items) if j%2 != 0]
                instance_n = f"obj_{i}"
                tile_n = fn.split('_',1)[-1].split('.')[0]
                wsi_n = fn.split('_', 1)[0]
                fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
                width = np.array(x).max() - np.array(x).min()
                height = np.array(y).max() - np.array(y).min()

                # fill df:
                info_dict = {'class_n':class_n, 'width':round((width), 4), 'height':round(height,4), 
                            'box_area':round(width*height, 4), 'obj_n':instance_n, 'fold':fold, 'tile':tile_n, 
                            'wsi':wsi_n, 'fn':fn.split('.')[0]}
                df.loc[i] = pd.Series(info_dict)

                i += 1
                
        # add also empty:
        self.df_instances = df
        self._add_empty2df()

        # print
        sel_col = [col for col in col_names if "wsi" not in col and 'fn' not in col]
        self.log.info(f"{self._class_name}.{'_get_instances_df'}: Df_instances:\n{self.df_instances[sel_col].head(3)}.")

        return df


    def _add_empty2df(self) -> None: 
        """ Helper function for _get_tiles_df. Adds to the dataframe also empty tiles."""

        _, empty = self._get_empty_images()

        # add empty tiles as new rows to self.df_instances:
        i = len(self.df_instances)
        for file in tqdm(empty, desc = 'Scanning empty tiles'):
            fn = os.path.basename(file)
            wsi_n = fn.split('_', 1)[0]
            fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
            info_dict = {'class_n':np.nan, 'width':np.nan, 'height':np.nan, 
                        'area':np.nan, 'obj_n':np.nan, 'fold':fold, 
                        'tile':tile_n,'wsi':{wsi_n}} # all empty values set to nan
            tile_n = fn.split('_',1)[-1].split('.')[0]
            self.df_instances.loc[i] = pd.Series(info_dict)

            i+=1

        return 
        



    def _get_tiles_df(self): 
        """ Creates a dataframe for tiles. """

        _, empty = self._get_empty_images()

        # create empty df:
        col_names = ['class_n', 'obj_n', 'tile', 'fold', 'wsi', 'fn' ]
        df = pd.DataFrame(columns=col_names)

        # open all labels: 
        label_f = self.data['tile_labels']
        i = 0
        for file in tqdm(label_f, desc = 'Creating tile_df'):
            
            # read label:
            with open(file, 'r') as f:
                rows = f.readlines()
            assert len(rows) <= 30, self.log.warning(f"❗️ Warning: File label has more than 30 instances. Maybe redundancy? ")
            # get items:
            class_n = 'glomerulus'
            rows = [row.replace('\n', '') for row in rows]
            for row in rows:
                items = row.split(' ')
                class_n = 'glomerulus' if int(float(items[0])) == 0 else class_n
            # fill df:
            fn = os.path.basename(file)
            n_objs = len(rows)
            tile_n = fn.split('_',1)[-1].split('.')[0]
            wsi_n = fn
            fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
            info_dict = {'class_n':class_n, 'obj_n':n_objs, 'fold':fold, 'tile':tile_n, 
                            'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
            df.loc[i] = pd.Series(info_dict)

            i += 1
        
        # adding also all empty tiles: 
        for file in empty:
            class_n = 'empty'
            n_objs = 0
            tile_n = fn.split('_',1)[-1].split('.')[0]
            wsi_n = fn.split('_', 1)[0]
            fold = os.path.split(os.path.split(os.path.dirname(file))[0])[1]
            info_dict = {'class_n':class_n, 'obj_n':n_objs, 'fold':fold, 
                            'tile':tile_n,'wsi':{wsi_n}, 'fn':{fn.split('.')[0]}}
            df.loc[i] = pd.Series(info_dict)
            i += 1
        
        # print
        sel_col = [col for col in col_names if "wsi" not in col and 'fn' not in col]
        self.log.info(f"{self._class_name}.{'_get_instances_df'}: Df_tiles:\n{df[sel_col].head(3)}.")

        return df
    

    def _get_ngloms_inslide(self, wsi_fp:str) -> int: 
        """ Computes number of glomeruli within a given slide. """

        assert os.path.isfile(wsi_fp), f"'wsi_fp':{wsi_fp} is not a valid filepath."

        # open wsi label and read n objects:
        with open(wsi_fp, mode='r') as f: 
            data = json.load(f)
        ngloms = len(data)
        self.log.info(f"{self._class_name}.{'_get_ngloms_inslide'}: found {ngloms} gloms in {os.path.basename(wsi_fp)}. ")
        
        return ngloms     
    

    def _get_ngloms_dataset(self) -> None:
        
        assert hasattr(self, 'data'), self.log.error(AttributeError(f"{self._class_name}.{'_get_ngloms_dataset'}: {self} doesn't have attribute 'data'."))
        
        annotations = self.data['wsi_labels']
        tot = 0
        for json_file in annotations: 
            with open(json_file, mode='r') as f: 
                data = json.load(f)
            tot += len(data)
        
        self.log.info(f"{self._class_name}.{'_get_ngloms_dataset'}: tot gloms in dataset: {tot}. ")

        return

    

    def __call__(self) -> None:

        self.data = self._get_data()
        self.df_instances = self._get_instances_df()
        self.df_tiles = self._get_tiles_df()
        self._get_class_freq()
        self._get_ngloms_dataset()
        self._get_ngloms_inslide(self.data['wsi_labels'][0])
        self._get_unique_labels()

        return


def test_profilerhubmap(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    data_root = '/Users/marco/helical_tests/test_hubmap_manager/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    wsi_images_like = '*.tif'
    wsi_labels_like = '*.json'
    tile_images_like = '*.png'
    tile_labels_like = '*.txt'
    n_classes = 1

    profiler = ProfilerHubmap(data_root=data_root, 
                            wsi_images_like = wsi_images_like, 
                            wsi_labels_like = wsi_labels_like,
                            tile_images_like = tile_images_like,
                            tile_labels_like = tile_labels_like,
                            n_classes=n_classes)

    profiler()
    return 


if __name__ == '__main__': 
    test_profilerhubmap()



