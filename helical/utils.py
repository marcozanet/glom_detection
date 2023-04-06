import os
from glob import glob
import random
from PIL import Image

def get_image_size(tile_fold:str):

    assert os.path.isdir(tile_fold), f"tile_fold:{tile_fold} is not a valid dirpath."
    images = glob(os.path.join(tile_fold, 'train', 'images', '*.png'))
    file = random.choice(images)
    image = Image.open(file)
    image_size = image.size

    return image_size[0]

def test_get_image_size(): 

    fp = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection/tiles'
    file = get_image_size(fp)
    image = Image.open(file)
    print(image.size)

    return

if __name__ == "__main__": 
    test_get_image_size()