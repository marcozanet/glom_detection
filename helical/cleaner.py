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

        print(len(unpaired_labels))
        print(unpaired_labels)

        for file in tqdm(unpaired_labels, desc = "Removing label"):
            os.remove(file)
        
        # super().__init__()

        return
    
    def _remove_perc_(self):

        tile_labels = self.data['tile_labels']           
        full, empty = self._get_empty_images()
        num_empty = len(empty)
        num_full = len(full)
        print(f"empty:{num_empty}, full:{num_full}")

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

        print(f"empty:{num_empty-len(img_empty2del)}, full:{num_full}")

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
        
        return 



def test_Cleaner():
    system = 'mac'
    safe_copy = True
    data_root = '/Users/marco/Downloads/try_train/detection' if system == 'mac' else r'C:\marco\biopsies\muw\detection'
    cleaner = Cleaner(data_root=data_root, safe_copy=safe_copy)
    cleaner()

    return


if __name__ == '__main__':
    test_Cleaner()