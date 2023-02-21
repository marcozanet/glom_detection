
import os
from glob import glob
from skimage import transform, io
from tqdm import tqdm
import numpy as np
import shutil
from typing import List, Tuple

class Resizer():

    def __init__(self,
                image_dir:str,
                out_shape:Tuple[int],
                copy_dataset:bool = False,
                verbose:bool = False ) -> None:

        """ image_dir: folder containing images.
            out_shape: shape of the desired output (height, width, without channels).
            copy_dataset:   if True -> copies all folders and files within image_dir and saves them into a new folder
                                named 'resized_<output_shape[0]> and replaces the resized images.
                            if False: saves resized images in a new folder named 'resized_<output_shape[0]> in the image_dir. 
            verbose: if True provides additional prints and info. """

        assert os.path.isdir(image_dir), f"image_dir:{image_dir} is not a valid dirpath."
        assert isinstance(verbose, bool), f"verbose:{verbose} should be boolean."
        assert isinstance(out_shape, tuple) and all([isinstance(val, int) for val in out_shape]), f"'out_shape':{out_shape} should be a tuple of int values."
        assert isinstance(copy_dataset, bool), f"copy_dataset:{copy_dataset} should be boolean."

        self.image_dir = image_dir
        self.verbose = verbose
        self.out_shape = out_shape
        self.copy_dataset = copy_dataset


        return
    

    def _get_files(self):

        files = glob(os.path.join(self.image_dir, f"*.png"))

        if len(files) == 0: # maybe it was a root folder -> look into train/val/test:
            files = glob(os.path.join(self.image_dir, '*', "images", f"*.png")) # only images, not labels
        
        files = [file for file in files if "DS" not in file] # skip temporary files
        if self.verbose is True:
            print(f"🔍 Found: {len(files)} files.")

        return files
    

    def _create_dirs(self) -> None:

        # prepare for saving
        if self.copy_dataset is False:
            save_dir = os.path.join(self.image_dir, f"resized_{self.out_shape[0]}")
            os.makedirs(save_dir, exist_ok=True)
            self.save_dir = save_dir
        else: #copy entire tree except images
            print(os.path.dirname(self.image_dir))
            new_tree = os.path.join(os.path.dirname(self.image_dir), f"tiles_{self.out_shape[0]}")
            
            if os.path.isdir(new_tree):
                print(f"Tree already existing. Skipping copy.")

            else:
                shutil.copytree(src = self.image_dir, dst=new_tree, ignore = shutil.ignore_patterns("*.png"))

            self.save_dir = new_tree

        return 


    def _resize(self, files: List[str]) -> None:

        for file in tqdm(files, desc = "Resizing"):
            image = io.imread(file)
            assert len(image.shape) == 3, f"Image shape:{image.shape} should be [w,h,ch]."
            w,h,ch = image.shape
            if self.out_shape[0] == w and self.out_shape[1] == h: 
                continue    # skip if already tiled
            out_shape = (self.out_shape[0], self.out_shape[1], ch)
            out_image = transform.resize(image, output_shape=out_shape) 
            out_image *= 255
            out_image = out_image.astype(np.uint8)

            # save
            self._save_file(fp = file, file=out_image)


        return
    
    def _save_file(self, fp:str, file:np.ndarray) -> None:

        assert isinstance(file, np.ndarray), f"'file' type is: {type(file)}, but should be np.ndarray."
        assert os.path.isdir(self.save_dir), f"save_dir:{self.save_dir} is not a valid dirpath."


        if self.copy_dataset is False: # saving in same folder
            dst_fp = os.path.join(self.save_dir, os.path.basename(fp))
            io.imsave(fname = dst_fp, arr = file, check_contrast=False)
            # shutil.copy(src = src_fp, dst = dst_fp)
        else:
            old_tree = self.image_dir
            new_tree = self.save_dir
            # print(f"old tree: {old_tree}")
            # print(f"new tree: {new_tree}")
            dst_fp = fp.replace(old_tree, new_tree)
            # print(dst_fp)
            io.imsave(fname = dst_fp, arr = file, check_contrast=False)
            # raise NotImplementedError()

            

        return
    

    def __call__(self) -> None:
        files = self._get_files()
        self._create_dirs()
        self._resize(files)
        return
    

def test_resizer():
        system = 'windows'
        image_dir = '/Users/marco/Downloads/try_train/detection/tiles' if system == 'mac' else r'D:\marco\datasets\muw\tiles_512'
        out_shape = (512,512)
        verbose = True
        copy_dataset = True
        resizer = Resizer(image_dir=image_dir, out_shape=out_shape, verbose=verbose, copy_dataset=copy_dataset)
        resizer()

        return

if __name__ == "__main__":
    test_resizer()