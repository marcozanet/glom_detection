
import sys
sys.path.append("/Users/marco/yolo/code")
import unittest

# from helical.utils.hey import print_hey
from helical.dataset import GlomDataset
from helical.dataloader import get_loaders

TRAIN_IMG_DIR = '/Users/marco/Downloads/folder_random/segmentation/train/images'
VAL_IMG_DIR = '/Users/marco/Downloads/folder_random/segmentation/val/images'
TEST_IMG_DIR = '/Users/marco/Downloads/folder_random/segmentation/test/images'
RESIZE = False
CLASSES = 4
MAPPING = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2, (255, 255, 255): 3}
BATCH = 2
NUM_WORKERS = 8


def test_GlomDatatset():


    trainset = GlomDataset(img_dir = TRAIN_IMG_DIR, 
                           resize = RESIZE, 
                           classes = CLASSES, 
                           mapping = MAPPING)
    trainset.__getitem__(idx = 0)
    trainset.__getitem__(idx = 1)
    trainset.__getitem__(idx = 2)
    trainset.__getitem__(idx = 3)
    return


def test_get_loaders():

    trainloader, valloader, testloader = get_loaders(train_img_dir = TRAIN_IMG_DIR, 
                                                    val_img_dir = VAL_IMG_DIR, 
                                                    test_img_dir = TEST_IMG_DIR, 
                                                    batch = BATCH, 
                                                    num_workers = NUM_WORKERS, 
                                                    resize = RESIZE, 
                                                    classes = CLASSES,
                                                    mapping = MAPPING) 
    
    return



if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # NON SO PERCHE' DA' SOLO 0 TEST RAN
    # # test_GlomDatatset()
    # test_get_loaders()
