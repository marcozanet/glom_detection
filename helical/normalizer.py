import torch
from torchvision import transforms
import torchstain
import cv2
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Tuple
from glob import glob
import random
from tqdm import tqdm


class Normalizer():

    def __init__(self, 
                target_path:str, 
                to_transform_path:str, 
                save_folder:str = None,
                show: bool = False,
                verbose:bool = False) -> None:

        save_folder = os.path.dirname(to_transform_path) if save_folder is None else save_folder
        assert os.path.isfile(target_path), f"'target_path':{target_path} is not a valid filepath."
        assert os.path.isfile(to_transform_path) or os.path.isdir(to_transform_path) , f"'to_transform_path':{to_transform_path} should be either a valid dirpath or a valid filepath."
        assert os.path.isdir(save_folder) , f"'save_folder':{save_folder} should be either a valid dirpath."
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."
        assert isinstance(show, bool), f"'show' should be a boolean."

        self.target_path = target_path
        self.to_transform_path = to_transform_path
        self.save_folder = (os.path.join(save_folder, 'normalized'))
        self.show = show
        self.verbose = verbose
        self.image_format = ".png"

        return
    

    def _normalize_file(self, file=str) -> np.ndarray:
        """ Stain normalizes a file. """

        # old_to_transform = io.imread(self.to_transform_path)
        # target_skimage = io.imread(self.target_path)

        target = cv2.cvtColor(cv2.imread(self.target_path), cv2.COLOR_BGR2RGB)
        to_transform = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])

        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))

        t_to_transform = T(to_transform)
        norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)

        # save: 
        norm = norm.numpy().astype(np.uint8)
        norm_fp = os.path.join(self.save_folder, os.path.basename(file))
        io.imsave(fname=norm_fp, arr= norm)
    
        return norm_fp
    

    def _normalize_folder(self) -> None:

        # 1) get images in folder 
        folder = self.to_transform_path
        images = glob(os.path.join(folder, f"*{self.image_format}"))

        # 2) normalize all images
        for image in tqdm(images, desc= "Normalizing"): 
            # normalize
            self._normalize_file(image)
        
        return 

    
    def _show_file(self, new_image:str) -> None:

        new_image = io.imread(new_image)
        old_image = io.imread(self.to_transform_path)
        target = io.imread(self.target_path)

        # Target image
        plt.figure()
        io.imshow(target)
        plt.title(label = f"Target image")
        plt.show()
        
        # Old image
        plt.figure()
        io.imshow(old_image)
        plt.title(label = f"Old image")
        plt.show()

        # New
        plt.figure()
        io.imshow(new_image)
        plt.title(label = f"New image")
        plt.show()

        return
    

    def _show_folder(self) -> None:
        

        # 1) get new_images: 
        new_images = glob(os.path.join(self.save_folder, f"*{self.image_format}"))
        # 2) pick random new_images to show: 
        show_images = random.sample(new_images, k = 5)

        # 3) show them in subplots
        K = 5
        fig, axs = plt.subplots(nrows = K, ncols = 2, figsize = (10,20), sharey=True)
        for i, new_fp in enumerate(show_images):
            new_fn = os.path.basename(new_fp)
            old_fp = os.path.join(self.to_transform_path, new_fn)
            assert os.path.isfile(old_fp), f"Old_image: {old_fp} does not exist."
            new_image = io.imread(new_fp)
            old_image = io.imread(old_fp)
            axs[i, 0].imshow(new_image)
            axs[i, 1].imshow(old_image)
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
        plt.suptitle(f"Stain normalized VS Normal", fontweight = 'bold')
        plt.tight_layout()



        return



    def __call__(self):

        # 0) create save folder:
        os.makedirs(self.save_folder, exist_ok=True)

        # 1) normalize: 
        if os.path.isfile(self.to_transform_path):
            norm_path = self._normalize_file(file=self.to_transform_path)
            # 2) show:
            if self.show is True:
                self._show_file(new_image=norm_path)
        else:
            self._normalize_folder()
            # 2) show:
            if self.show is True:
                self._show_folder()
         
        return

def test_Normalizer(): 

    # print(" ########################    TEST file 1: ⏳    ########################")
    # target_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/201025745_09_SFOG_sample0_19_12.png'
    # to_transform_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/200701099_09_SFOG_sample0_22_28.png'
    # verbose = True 
    # show = True
    # normalizer = Normalizer(target_path=target_path, to_transform_path=to_transform_path, verbose = verbose, show = show)
    # normalizer()
    # print(" ########################    TEST file 1: ✅    ########################")

    # print(" ########################    TEST file 2: ⏳    ########################")
    # target_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/201025745_09_SFOG_sample0_19_12.png'
    # to_transform_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/200822954_09_SFOG_sample0_2_6.png'
    # verbose = True 
    # show = True
    # normalizer = Normalizer(target_path=target_path, to_transform_path=to_transform_path, verbose = verbose, show = show)
    # normalizer()
    # print(" ########################    TEST file 2: ✅    ########################")

    print(" ########################    TEST folder: ⏳    ########################")
    target_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/200725057_09_SFOG_sample0_16_26.png'
    to_transform_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer'
    verbose = True 
    show = True
    save_folder = None
    normalizer = Normalizer(target_path=target_path, to_transform_path=to_transform_path, 
                            verbose = verbose, show = show, save_folder=save_folder)
    normalizer()
    print(" ########################    TEST folder: ✅    ########################")
    return

if __name__ == '__main__': 
    test_Normalizer()
