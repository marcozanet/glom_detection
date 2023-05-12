
from typing import Literal, Tuple
import os, shutil
from tqdm import tqdm
from glob import glob
import numpy as np
from converter_muw_new import ConverterMuW
from converter_hubmap import ConverterHubmap
from tiler_hubmap import TilerHubmap
from tiler_muw_new import TilerMuwDetection
from manager_base import ManagerBase



class ManagerDetectMuwSFOG(ManagerBase): 

    def __init__(self,
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        assert self.data_source == 'muw', self.log.error(ValueError(f"'data_source' is {self.data_source} but Manager used is 'ManagerMUW'"))
        self.data_source == "muw"

        return

    
    def _rename_mrxsgson2gson(self):

        files = [os.path.join(self.src_root,file) for file in os.listdir(self.src_root) if '.mrxs.gson' in file]
        old_new_names = [(file, file.replace('.mrxs.gson', '.gson')) for file in files ]
        for old_fp, new_fp in old_new_names: 
            os.rename(old_fp, new_fp)

        return


    def _tile_folder(self, dataset:Literal['train', 'val', 'test']):
        """ Tiles a single folder"""

        class_name = self.__class__.__name__
        func_name = '_tile_folder'
        slides_labels_folder = os.path.join(self.wsi_dir, dataset, 'labels')
        save_folder_labels = os.path.join(self.tiles_dir, dataset)
        save_folder_images = os.path.join(self.tiles_dir, dataset)

        # check that fold is not empty: 
        if len(os.listdir(slides_labels_folder)) == 0:
            self.log.warn(f"{class_name}.{func_name}: {dataset} fold is empty. Skipping.")
            return

        # 1) convert annotations to yolo format:
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ⏳    ########################")
        if self.data_source == 'muw':
            converter = ConverterMuW(folder = slides_labels_folder, 
                                     stain = self.stain,
                                     multiple_samples = self.multiple_samples,
                                    convert_from='gson_wsi_mask',  
                                    convert_to='txt_wsi_bboxes',
                                    save_folder= slides_labels_folder, 
                                    level = self.tiling_level,
                                    verbose=self.verbose)
        elif self.data_source == 'hubmap':
            converter = ConverterHubmap(folder = slides_labels_folder,
                                        multiple_samples = self.multiple_samples, 
                                        stain = self.stain,
                                        convert_from='json_wsi_mask',  
                                        convert_to='txt_wsi_bboxes',
                                        save_folder= slides_labels_folder, 
                                        level = self.tiling_level,
                                        verbose=self.verbose)
        converter()
        self.log.info(f"{class_name}.{func_name}: ######################## CONVERTING ANNOTATIONS: ✅    ########################")

        # 2) tile images:
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")
        if self.data_source == 'muw':
            tiler = TilerMuwDetection(folder = slides_labels_folder, 
                                    tile_shape= self.tiling_shape, 
                                    step=self.tiling_step, 
                                    save_root= save_folder_images, 
                                    level = self.tiling_level,
                                    show = self.tiling_show,
                                    verbose = self.verbose,
                                    resize = self.resize,
                                    multiple_samples = self.multiple_samples)
            self.tiler = tiler
        elif self.data_source == 'hubmap': 
            print(f"alling tiler hubmap")
            tiler = TilerHubmap(folder = slides_labels_folder, 
                                tile_shape= self.tiling_shape, 
                                step=self.tiling_step, 
                                save_root= save_folder_images, 
                                level = self.tiling_level,
                                show = self.tiling_show,
                                verbose = self.verbose)
        target_format = 'tif'
        tiler(target_format=target_format)
        self.log.info(f"{class_name}.{func_name}: ######################## TILING IMAGES: ⏳    ########################")

        return
    

    def _segmentation2detection(self): 

        # get all label files:
        labels = glob(os.path.join(self.dst_root, self.task, 'tiles', '*', 'labels', f"*.txt"))
        labels = [file for file in labels if 'DS' not in file]
        assert len(labels)>0, f"'labels' like: {os.path.join(self.dst_root, self.task, 'tiles', '*', 'labels',  f'*.txt')} is empty."

        # loop through labels and get bboxes:
        for file in tqdm(labels, desc='transforming segm labels to bboxes'): 
            with open(file, 'r') as f: # read file
                text = f.readlines()
            new_text=''
            for row in text: # each row = glom vertices
                row = row.replace(' /n', '')
                items = row.split(sep = ' ')
                class_n = int(float(items[0]))
                items = items[1:]
                x = [el for (j,el) in enumerate(items) if j%2 == 0]
                x = np.array([float(el) for el in x])
                y = [el for (j,el) in enumerate(items) if j%2 != 0]
                y = np.array([float(el) for el in y])
                x_min, x_max = str(x.min()), str(x.max())
                y_min, y_max = str(y.min()), str(y.max())
                new_text += str(class_n)
                new_text += f" {x_min} {y_min} {x_max} {y_min} {x_max} {y_max} {x_min} {y_max}"
                new_text += '\n'
            
            with open(file, 'w') as f: # read file
                f.writelines(new_text)

        # use self.tiler to show new images:
        print('plotting images')
        self.tiler.test_show_image_labels()
        print('plotting images done. ')

        return


    def __call__(self) -> None:

        self._rename_tiff2tif()
        self._rename_mrxsgson2gson()
        # 1) create tiles branch
        self._make_tiles_branch()
        # 1) split data
        self._split_data()
        # 2) prepare for tiling 
        self._move_slides_forth()
        # 3) tile images and labels:
        self.tile_dataset()
        # 4) move slides back 
        self._move_slides_back()
        # 5) clean dataset, e.g. 
        self._clean_muw_dataset()

        if self.task == 'detection':
            self._segmentation2detection()
            print('segm 2 detection done')

        return
    



