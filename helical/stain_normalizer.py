


import torch
from torchvision import transforms
import torchstain
import cv2
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np


class Normalizer():


    def __init__(self, 
                target_path:str, 
                to_transform_path:str, 
                verbose:bool = False) -> None:

        assert os.path.isfile(target_path), f"'target_path':{target_path} is not a valid filepath."
        assert os.path.isfile(to_transform_path), f"'to_transform_path':{to_transform_path} is not a valid filepath."
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."

        return
    

    def normalize_file(self, target_path:str, to_transform_path:str):
        """ Stain normalizes a file. """

        old_to_transform = io.imread(to_transform_path)
        target_skimage = io.imread(target_path)

        target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
        to_transform = cv2.cvtColor(cv2.imread(to_transform_path), cv2.COLOR_BGR2RGB)

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])

        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))

        t_to_transform = T(to_transform)
        norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)
    
        return
    
    def _show_(self, old_image: np.array, new_image: torch.Tensor):


        print(type(norm))
        norm = norm.numpy()

        io.imshow(target_skimage)
        plt.show()
        io.imshow(old_to_transform)
        plt.show()
        io.imshow(norm)
        plt.show()

        return

if __name__ == '__main__': 
    target_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/201025745_09_SFOG_sample0_19_12.png'
    to_transform_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/200701099_09_SFOG_sample0_22_28.png'
    stain_normalize(target_path=target_path, to_transform_path=to_transform_path)
    to_transform_path = '/Users/marco/Downloads/test_folders/test_stainnormalizer/200822954_09_SFOG_sample0_2_6.png'
    stain_normalize(target_path=target_path, to_transform_path=to_transform_path)
