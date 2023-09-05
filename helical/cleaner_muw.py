### cleans dataset from e.g. duplicates and from having too many empty images -> removes majority of them.

from profiler_muw import ProfilerMUW
import os, shutil
from tqdm import tqdm
import random 
import sys
from decorators import log_start_finish
from utils import copy_tree



class CleanerMuw(ProfilerMUW):

    def __init__(self, 
                empty_perc:float = 0.1,  
                safe_copy:bool = True,
                remove_classes: list = None,
                ignore_classes: list = None,
                *args, 
                **kwargs):
            
        super().__init__(*args, **kwargs)

        self.empty_perc = empty_perc
        self.safe_copy = safe_copy
        self.remove_classes = remove_classes
        self.ignore_classes = ignore_classes
        self.data = self._get_data()
        return
    

    def _del_unpaired_labels(self):
        """ Removes labels that don't have a corresponding image (the opposite is fine). """

        tile_images = self.data['tile_images']
        tile_labels = self.data['tile_labels']  

        rename_img2lbl = lambda fp_img: os.path.join(os.path.dirname(fp_img).replace('images', 'labels'), os.path.basename(fp_img).replace(self.tiles_image_format, self.tiles_label_format))
        rename_lbl2img = lambda fp_lbl: os.path.join(os.path.dirname(fp_lbl).replace('labels', 'images'), os.path.basename(fp_lbl).replace(self.tiles_label_format,self.tiles_image_format))
        
        if self.verbose is True:
            self.log.info(f"looking for labels like {rename_img2lbl(tile_images[0])}")
        unpaired_labels = [file for file in tile_labels if rename_lbl2img(file) not in tile_images]

        def do():
            n_removed=0
            for file in tqdm(unpaired_labels, desc = "Removing unpaired label"):
                assert os.path.isfile(file), self.log.error(f"{self._class_name}.{'_replacing_class'}: AssertionError:'file':{file} is not a valid filepath.")
                # self.log.info(f"assert passed")
                os.remove(file)
                n_removed+=1
            
            # update self.data: 
            self.log.info(f"{self._class_name}.{'_del_unpaired_labels'}: deleted {n_removed} labels without matching image. Updating self.data. ")
            self.data = self._get_data()
            
            return
        
        do()

        return
    


    def _remove_class_(self, class_num:int=3) -> None:
        """ Gets rid of a class -> deletes corresponding rows from label files."""

        # make sure class to remove is one of unique labels: 
        unique_classes = self._get_unique_labels()
        assert str(class_num) in unique_classes, self.log.error(f"{self._class_name}.{'_replacing_class'}: AssertionError:'class':{class_num} not in unique classes: {unique_classes}")

            
        # look into each label file:
        for label_fp in tqdm(self.data['tile_labels'], desc = f" Removing class '{class_num}' from labels"):

            assert os.path.isfile(label_fp), self.log.error(f"'label_fp':{label_fp} is not a valid filepath.")
            
            # pop out the desired class:
            found = False
            with open(label_fp, mode ='r') as f:
                old_rows = f.readlines()
                new_rows = old_rows.copy()
                for i, row in enumerate(old_rows):
                    if row[0] == str(class_num):
                        new_rows.remove(row)
                        found = True

            # overwrite label file:
            if found is True:
                with open(label_fp, 'w') as f:
                    f.writelines(new_rows)
                    if self.verbose is True:
                        print(f"Removed class.")
        
        # final check:
        self.data = self._get_data()
        unique_classes = self._get_unique_labels()
        assert str(class_num) not in unique_classes, self.log.error(f"Removed class {class_num} still appears in 'unique_classes':{unique_classes}")
        
        # update self.data: 
        self.data = self._get_data()


        return


    def _replacing_class(self, class_old:int = 2, class_new:int = 1) -> None:
        """ Replaces class_old with class_new, e.g. class_old=2, class_new= 1: old_label = {0,2,3} -> {0,1,3} merging 1 and 2 -> {0, 1 (=ex 1,2), 2} """

            # at the beginning {0:healthy, 1: NA, 2:unhealthy, 3:tissue}
        unique_classes = self._get_unique_labels(verbose=True)
        assert str(class_old) in unique_classes, self.log.error(f"{self._class_name}.{'_replacing_class'}: AssertionError:'class_old':{class_old} not found in unique classes: {unique_classes}")
        assert str(class_new) not in unique_classes, self.log.error(f"{self._class_name}.{'_replacing_class'}: AssertionError:'class_new':{class_new} already found in unique classes: {unique_classes}")
        assert class_old != class_new, self.log.error(f"{self._class_name}.{'_replacing_class'}: AssertionError:class_old and class_new can't be the same class.")

        @log_start_finish(class_name=self.__class__.__name__, func_name='_replacing_class', msg = f" Replacing class:{class_old} with class:{class_new}" )
        def do():           

            # look into each label file:
            for label_fp in tqdm(self.data['tile_labels'], desc = f"Replacing class:{class_old} with class:{class_new}"):
                
                # get rows from file:
                with open(label_fp, mode ='r') as f:
                    old_rows = f.readlines()
                
                # replace class_max with class_min:
                new_rows = []
                found = False
                for row in old_rows:
                    if row[0] == str(class_old):
                        found = True
                        row = str(class_new) + row[1:]
                    new_rows.append(row)

                assert len(old_rows) == len(new_rows), f"Old label and new label don't have the same length: old_label(len:{len(old_rows)}={old_rows}. \nnew_label(len:{len(new_rows)}={new_rows})"
                
                if found is True:
                    # overwrite label file:
                    with open(label_fp, 'w') as f:
                        f.writelines(new_rows)
            
            # final check:
            unique_classes = self._get_unique_labels()
            assert str(class_old) not in unique_classes, f"Replaced class:{class_old} still appears in 'unique_classes':{unique_classes}"
            self.log.info(f"{self.__class__.__name__}.{'_replacing_class'}: New unique classes: unique_classes = {self._get_unique_labels()} ")

            # update self.data: 
            self.data = self._get_data()

            return

        do()

        return
    
    
    def _assign_NA_randomly(self, NA_class:int = 1):
        """ Randomly assings NA values to either class 0 or class 2"""

        unique_classes = self._get_unique_labels(verbose=True)
        assert NA_class in [int(el) for el in unique_classes], f"Merging NA_class:{NA_class}, but unique classes are:{unique_classes}"
        assert str(NA_class) in unique_classes, f"'NA_class':{NA_class} not found in unique classes: {unique_classes}"

        @log_start_finish(class_name=self.__class__.__name__, func_name='_replacing_class', msg = f" Assigning randomly class:{NA_class} to either class:0 or class:2." )
        def do():  
            # look into each label file:
            for label_fp in tqdm(self.data['tile_labels'], desc = f"Replacing class:{NA_class} with random 0 or 1"):
                
                # get rows from file:
                with open(label_fp, mode ='r') as f:
                    old_rows = f.readlines()
                
                # replace class_max with class_min:
                new_rows = []
                found = False
                for row in old_rows:
                    if row[0] == str(NA_class):
                        found = True
                        # print(f"class max found : {row[0]}")
                        class_new = random.choice([0,2])
                        row = str(class_new) + row[1:]
                        # print(row)
                    # else:
                    #     found = False
                    new_rows.append(row)

                assert len(old_rows) == len(new_rows), f"Old label and new label don't have the same length: old_label(len:{len(old_rows)}={old_rows}. \nnew_label(len:{len(new_rows)}={new_rows})"
                
                if found is True:

                    # overwrite label file:
                    with open(label_fp, 'w') as f:
                        f.writelines(new_rows)
            
            # final check:
            unique_classes = self._get_unique_labels()
            assert str(NA_class) not in unique_classes, f"Replaced class:{NA_class} still appears in 'unique_classes':{unique_classes}"

            if self.verbose is True:
                print(f"New unique classes: unique_classes = {self._get_unique_labels()} ")

            # update self.data: 
            self.data = self._get_data()

            return
        
        do()

        return


    def _remove_small_objs(self, w_thr:float=0.15, h_thr:float=0.15) -> None: 
        """ Removes label objects where area < thr """

        @log_start_finish(class_name=self.__class__.__name__, func_name='_remove_small_objs', msg = f" Removing objs<{(w_thr, h_thr)}." )
        def do():  
            # look into each label file:
            n_inst_pre = 0
            n_inst_post = 0
            for label_fp in tqdm(self.data['tile_labels'], desc = f"Removing small annotations:"):
                
                # get rows from file:
                with open(label_fp, mode ='r') as f:
                    old_rows = f.readlines()
                
                # add accepted objs to a new label text:
                new_rows = []
                for row in old_rows:
                    n_inst_pre += 1
                    items = row.split(' ')
                    w, h = [float(el) for el in items[3:]]
                    if w >= w_thr and h >= h_thr: 
                        n_inst_post += 1
                        new_rows.append(row)

                # overwrite label file:
                with open(label_fp, 'w') as f:
                    f.writelines(new_rows)

            if self.verbose is True:      
                print(f"✅ Instances before removal:{n_inst_pre}, post removal:{n_inst_post}")
            
            # update self.data: 
            self.data = self._get_data()
            return
        do()
        
        return



    def _merge_classes(self, class_1:int, class_2:int) -> None:
        """ Merges 2 classes together, e.g. {0, 1, 2, 3} -> merging 1 and 2 -> {0, 1 (=ex 1,2), 2} """

        unique_classes = self._get_unique_labels(verbose=True)
        assert str(class_1) in unique_classes, f"'class_1':{class_1} not found in unique classes: {unique_classes}"
        assert str(class_2) in unique_classes, f"'class_2':{class_2} not found in unique classes: {unique_classes}"
        assert class_1 != class_2, f"class_1 and class_2 can't be the same class."

        @log_start_finish(class_name=self.__class__.__name__, func_name='_merge_classes', msg = f" Merging class:{class_1} and class:{class_2}" )
        def do(): 

            if class_1 > class_2:
                class_min = class_2
                class_max = class_1 
            else:
                class_min = class_1 
                class_max = class_2


            # look into each label file:
            for label_fp in tqdm(self.data['tile_labels'], desc = f"Merging classes {class_1} and {class_2}"):
                
                # get rows from file:
                with open(label_fp, mode ='r') as f:
                    old_rows = f.readlines()
                
                # replace class_max with class_min:
                new_rows = []
                found = False
                for row in old_rows:
                    if row[0] == str(class_max):
                        found = True
                        # print(f"class max found : {row[0]}")
                        row = str(class_min) + row[1:]
                        # print(row)
                    # else:
                    #     found = False
                    new_rows.append(row)

                assert len(old_rows) == len(new_rows), f"Old label and new label don't have the same length: old_label(len:{len(old_rows)}={old_rows}. \nnew_label(len:{len(new_rows)}={new_rows})"
                
                if found is True:
                    # print(old_rows)
                    # print(new_rows)

                    # overwrite label file:
                    with open(label_fp, 'w') as f:
                        f.writelines(new_rows)
            
            # final check:
            unique_classes = self._get_unique_labels()
            assert str(class_max) not in unique_classes, f"Merged class:{class_max} still appears in 'unique_classes':{unique_classes}"

            if self.verbose is True:
                print(f"✅ New unique classes: unique_classes = {self._get_unique_labels()} ")

            # update self.data: 
            self.data = self._get_data()

            return
        
        do()

        return

    

    def _del_redundant_labels(self):
        """ Removes redundancy in labels, e.g. same rows within label 
            files (happens due to the a+ file writing mode)."""

        for label_fp in tqdm(self.data['tile_labels'], desc = 'Deleting redundancies in label'):
            unique_lines= []
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                for row in rows:
                    if row not in unique_lines:
                        unique_lines.append(row)
            with open(label_fp, 'w') as f:
                f.writelines(unique_lines)
        
        # update self.data: 
        self.data = self._get_data()

        return

    
    def _remove_perc_(self):
        """ Removes empty images, so that remaining empty images are self.perc% of the total images."""

        filtered_data = self._get_only_train_val_files()
        tile_labels = filtered_data['tile_labels']           
        full, empty = self._get_empty_images(also_from_test=False)
        num_empty = len(empty)
        num_full = len(full)

        assert num_full > 0, self.log.error(f"{self._class_name}.{'_remove_perc_'}: no full image found. full images:{num_full}, empty images:{num_empty}")

        def do():
            # compute num images to del:
            tot = round(num_full / (1 - self.empty_perc))
            num_desired_empty = tot - num_full
            to_del = num_empty - num_desired_empty
            if to_del <= 0: 
                self.log.warning(f"{self._class_name}.{'_remove_perc_'}:❗️ Empty images ({num_empty}/{tot}) are already <= {self.empty_perc*100}% of tot images. Skipping removal of empty images.")
                return
            # select k random empty samples to del:
            img_empty2del = random.sample(empty, k = to_del)
            # check that no labels are deleted:
            lbl_empty2del = [os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format, self.tiles_label_format)) for file in img_empty2del]
            lbl_empty2del = [file for file in lbl_empty2del if file in tile_labels]
            assert len(lbl_empty2del) == 0, f"❌ some of the selected images for delete do have a corresponding label."

            if self.verbose is True:
                self.log.info(f"{self._class_name}.{'_remove_perc_'}: Deleting {len(img_empty2del)} images")
                self.log.info(f"{self._class_name}.{'_remove_perc_'}: Deleting {len(lbl_empty2del)} labels")

            for file in tqdm(img_empty2del, desc = "Removing empty images"):
                os.remove(file)
            
            self.log.info(f"{self._class_name}.{'_remove_perc_'}: Keeping {self.empty_perc*100}% of empty images. Empty images:{num_empty-len(img_empty2del)}, full images:{num_full}. Tot images:{num_empty-len(img_empty2del) + num_full}")

            return
        
        do()

        return
    
    def _remove_empty_labels(self):
        """ Reads through label files and deletes the empty ones. """

        tile_labels = self.data['tile_labels']  

        for file in tqdm(tile_labels, desc="Removing empty labels"): 
            
            #open file
            with open(file, 'r') as f:
                text = f.readlines()

            # check if empty and remove
            if len(text) == 0:
                if self.verbose is True:
                    print(f"Removing: {file}")
                os.remove(file)
        
        # update self.data: 
        self.data = self._get_data()

        return
    


    def _copy_tree(self) -> None:
        """ Copies folder tree if safe copy is True"""

        src = os.path.join(self.data_root, 'tiles')
        dst = os.path.join(os.path.dirname(self.data_root), 'safe_copy')
        if not os.path.isdir(dst):
            self.log.info(f"{self._class_name}.{'_copy_tree'}: Safe copying tree before modifying:")
            shutil.copytree(src = src, dst = dst)
        else:
            self.log.info(f"{self._class_name}.{'_copy_tree'}: Safe copy of the data tree already existing. Skipping.")

        return
    
    def _clean_muw(self): 

        if self.safe_copy is True:
            self._copy_tree()
        # 1) delete labels where image doesn't exist
        self._del_unpaired_labels()
        # 2) remove label redundancies
        self._del_redundant_labels()
        # # 5) removes tissue class
        # self._remove_class_(class_num=3)
        # # 6) assign randomly the NA class (int=1) to either class 0 or 2:
        self._assign_NA_randomly()
        # # 7) replace class 2 with class 1 ({0:healthy, 1:NA, 2:unhealthy} -> {0:healthy, 2:unhealthy})
        self._replacing_class(class_old=2, class_new=1)
        # 8) removing empty images to only have empty_perc of images being empty
        self._remove_perc_()
        

        return
    
    def _clean_generic(self): 

        func_n = self._clean_generic.__name__

        if self.safe_copy is True:
            self._copy_tree()
        # 1) delete labels where image doesn't exist
        self._del_unpaired_labels()
        # 2) remove label redundancies
        self._del_redundant_labels()
        # 8) removing empty images to only have empty_perc of images being empty
        self._remove_perc_()

        if self.remove_classes is not None:
            for clss in self.remove_classes:
                self._remove_class_(class_num=clss)
        
        if self.ignore_classes is not None:
            self.format_msg(f"Copying tree to {os.path.join(self.data_root, 'temp')} with {self.tile_labels_like}", func_n=func_n, type='warning')
            # 1) Copy labels in a temp tree
            copy_tree(tree_path=self.data_root, 
                        dst_dir=os.path.join(self.data_root, 'temp'), 
                        keep_format="."+self.tiles_label_format)
            # 2) remove classes:
            for clss in self.ignore_classes:
                self._remove_class_(class_num=clss)
        

        return
    


    
    def __call__(self) -> None:

        self._clean_muw()



        return 



# def test_Cleaner():
#     import sys 
#     system = 'mac' if sys.platform == 'darwin' else 'windows'
    
#     safe_copy = False
#     data_root = '/Users/marco/converted_march'
#     # data_root = '/Users/marco/Downloads/train_20feb23/' if system == 'mac' else r'D:\marco\datasets\muw\detection'
#     cleaner = CleanerMuw(data_root=data_root, 
#                         safe_copy=safe_copy,
#                         wsi_images_like = '*.tif', 
#                         wsi_labels_like = '*.gson',
#                         tile_images_like = '*.png',
#                         tile_labels_like = '*.txt',)
#     cleaner()
#     return


# if __name__ == '__main__':
#     test_Cleaner()