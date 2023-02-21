### cleans dataset from e.g. duplicates and from having too many empty images -> removes majority of them.

from profiler import Profiler
from shutil import copytree
import os
from tqdm import tqdm
import random


class Cleaner(Profiler):

    def __init__(self, 
                empty_perc:float = 0.1,  
                safe_copy:bool = True,
                *args, 
                **kwargs):
            
        super().__init__(*args, **kwargs)

        self.empty_perc = empty_perc
        self.safe_copy = safe_copy

        return
    

    def _del_unpaired_labels(self):
        """ Removes labels that don't have a corresponding image (the opposite is fine). """
        tile_images = self.data['tile_images']
        tile_labels = self.data['tile_labels']        
        unpaired_labels = [file for file in tile_labels if os.path.join(os.path.dirname(file).replace('labels', 'images'), os.path.basename(file).replace(self.tiles_label_format, self.tiles_image_format)) not in tile_images]

        # print(len(unpaired_labels))
        # print(unpaired_labels)

        for file in tqdm(unpaired_labels, desc = "Removing unpaired label"):
            os.remove(file)
        
        # super().__init__()

        return


    def _remove_class_(self, class_num:int=3) -> None:
        """ Gets rid of a class -> deletes corresponding rows from label files."""

        # look into each label file:
        for label_fp in tqdm(self.data['tile_labels'], desc = f"Removing class '{class_num}' from labels"):
            
            # pop out the desired class:
            found = False
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                for i, row in enumerate(rows):
                    if row[0] == str(class_num):
                        print(f"rows before: {rows}")
                        rows.pop(i)
                        print(f"rows after: {rows}")
                        found = True
                    else:
                        found = False

            # overwrite label file:
            if found is True:
                with open(label_fp, 'w') as f:
                    f.writelines(rows)
                    print(f"Removed class.")
        
        # final check:
        unique_classes = self._get_unique_labels()
        assert str(class_num) not in unique_classes, f"Removed class still appears in 'unique_classes':{unique_classes}"


        return


    def _replacing_class(self, class_old:int = 2, class_new:int = 1) -> None:
        """ Replaces class_old with class_new, e.g. class_old=2, class_new= 1: old_label = {0,2,3} -> {0,1,3} merging 1 and 2 -> {0, 1 (=ex 1,2), 2} """

        unique_classes = self._get_unique_labels(verbose=True)
        assert str(class_old) in unique_classes, f"'class_old':{class_old} not found in unique classes: {unique_classes}"
        assert str(class_new) not in unique_classes, f"'class_new':{class_new} already found in unique classes: {unique_classes}"
        assert class_old != class_new, f"class_old and class_new can't be the same class."


        # look into each label file:
        for label_fp in tqdm(self.data['tile_labels'], desc = f"Replaceing class:{class_old} with class:{class_new}"):
            
            # get rows from file:
            with open(label_fp, mode ='r') as f:
                old_rows = f.readlines()
            
            # replace class_max with class_min:
            new_rows = []
            found = False
            for row in old_rows:
                if row[0] == str(class_old):
                    found = True
                    # print(f"class max found : {row[0]}")
                    row = str(class_new) + row[1:]
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
        assert str(class_old) not in unique_classes, f"Replaced class:{class_old} still appears in 'unique_classes':{unique_classes}"

        print(f"New unique classes: unique_classes = {self._get_unique_labels()} ")

        return


    def _merge_classes(self, class_1:int, class_2:int) -> None:
        """ Merges 2 classes together, e.g. {0, 1, 2, 3} -> merging 1 and 2 -> {0, 1 (=ex 1,2), 2} """

        unique_classes = self._get_unique_labels(verbose=True)
        assert str(class_1) in unique_classes, f"'class_1':{class_1} not found in unique classes: {unique_classes}"
        assert str(class_2) in unique_classes, f"'class_2':{class_2} not found in unique classes: {unique_classes}"
        assert class_1 != class_2, f"class_1 and class_2 can't be the same class."

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

        print(f"New unique classes: unique_classes = {self._get_unique_labels()} ")

        return

    
    def _del_redundant_labels(self):
        """ Removes redundancy in labels, e.g. same rows within label 
            files (happens due to the a+ file writing mode)."""

        for label_fp in tqdm(self.data['tile_labels'], desc = 'Deleting redundancies in label'):
            unique_lines= []
            with open(label_fp, mode ='r') as f:
                rows = f.readlines()
                # print(rows)
                for row in rows:
                    if row not in unique_lines:
                        unique_lines.append(row)
            with open(label_fp, 'w') as f:
                f.writelines(unique_lines)
        

        return
    


    
    def _remove_perc_(self):

        tile_labels = self.data['tile_labels']           
        full, empty = self._get_empty_images()
        num_empty = len(empty)
        num_full = len(full)
        print(f"empty:{num_empty}, full:{num_full}, tot:{num_empty + num_full}")

        # compute num images to del:
        tot = round(num_full / (1 - self.empty_perc))
        num_desired_empty = tot - num_full
        to_del = num_empty - num_desired_empty
        
        # select k random empty samples to del:
        img_empty2del = random.sample(empty, k = to_del)

        # check that no labels are deleted:
        lbl_empty2del = [os.path.join(os.path.dirname(file).replace('images', 'labels'), os.path.basename(file).replace(self.tiles_image_format, self.tiles_label_format)) for file in img_empty2del]
        lbl_empty2del = [file for file in lbl_empty2del if file in tile_labels]
        assert len(lbl_empty2del) == 0, f"❌ some of the selected images for delete do have a corresponding label."

        print(f"Deleting {len(img_empty2del)} images")
        print(f"Deleting {len(lbl_empty2del)} labels")

        for file in tqdm(img_empty2del, desc = "Removing empty images"):
            os.remove(file)

        print(f"empty:{num_empty-len(img_empty2del)}, full:{num_full}, tot:{num_empty-len(img_empty2del) + num_full}")

        return
    
    def _copy_tree(self) -> None:
        """ Copies folder tree if safe copy is True"""

        src = os.path.join(self.data_root, 'tiles')
        dst = os.path.join(os.path.dirname(self.data_root), 'safe_copy')
        if not os.path.isdir(dst):
            print(f"Safe copying tree before modifying:")
            copytree(src = src, dst = dst)
        else:
            print(f"Safe copy of the data tree already existing. Skipping.")

        return

    
    def __call__(self) -> None:

        if self.safe_copy is True:
            self._copy_tree()

        self._del_unpaired_labels()

        self._remove_perc_()

        self._del_redundant_labels()

        self._remove_class_(class_num=3)

        # NB DONT RUN WHEN THERE'S ONLY 0 AND 1 OR YOU'LL GET JUST ONE CLASS
        # self._merge_classes(class_1=0, class_2=1)

        self._replacing_class(class_old=2, class_new=1)


        return 



def test_Cleaner():
    system = 'mac'
    safe_copy = False
    data_root = '/Users/marco/Downloads/train_20feb23/' if system == 'mac' else r'C:\marco\biopsies\muw\detection'
    cleaner = Cleaner(data_root=data_root, safe_copy=safe_copy)
    cleaner()

    return


if __name__ == '__main__':
    test_Cleaner()