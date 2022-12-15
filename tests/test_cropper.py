import sys
sys.path.append("/Users/marco/yolo/code")
import unittest
from helical.cropping import Cropper


def test_Cropper():

    root = '/Users/marco/datasets/muw_exps'
    save_folder = '/Users/marco/datasets/muw_exps/segmentation/images'
    cropper = Cropper(root = root, 
                      save_folder = save_folder, 
                      image_shape=(4096, 4096),
                      percentile = 90)
    cropper()

    return



if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 