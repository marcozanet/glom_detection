import sys
sys.path.append("/Users/marco/yolo/code")
import unittest
from skimage import io, color
import numpy as np
from helical.utils import utils_image


def test_map_RGB2coloridx():

    mapping =  {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2, (255, 255, 255): 3}
    mask_fp = '/Users/marco/datasets/muw_exps/segmentation/train/masks/200104066_09_SFOG [x=4096,y=139264,w=4096,h=4096].png'
    mask = io.imread(mask_fp)

    mask = color.rgba2rgb(mask)*255 if mask.shape[2] == 4 else mask
    mask = np.zeros_like(mask)
    mask = utils_image.map_RGB2coloridx(mask = mask, mapping = mapping, classes = 4)

    return


def test_get_corners():

    file = '/Users/marco/datasets/muw_exps/segmentation/train/masks/200104066_09_SFOG [x=4096,y=139264,w=4096,h=4096].png'
    mapping =  {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2, (255, 255, 255): 3}
    image = io.imread(file)
    io.imshow(image)
    image = color.rgba2rgb(image)*255 if image.shape[2] == 4 else image
    image = utils_image.map_RGB2coloridx(mask=image, mapping=mapping, classes=4, one_hot=False)
    image = image.numpy()
    print(image.shape)
    # image = np.moveaxis(image, 0, -1)
    # image = color.rgb2gray(image)
    image = image * (255/3)
    image = image.astype(np.int32)
    print(np.unique(image))
    
    io.imshow(image)
    utils_image.get_corners(image=image)

    return


if __name__ == "__main__":
    test_get_corners()
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)