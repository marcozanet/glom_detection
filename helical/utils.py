import os, sys, shutil
from glob import glob
import random
from PIL import Image
import yaml 
from yaml import SafeLoader
import tqdm



def copy_tree(tree_path:str, dst_dir:str, keep_format:str)->None:
    """ Copies an entire tree structure without copying the contained files. """
    assert '.' in keep_format, f"copy_tree: 'keep_format' should contain ."
    ignore_files = lambda dir,files:[f for f in files if os.path.isfile(os.path.join(dir, f)) if keep_format not in f ] 
    shutil.copytree(tree_path, dst_dir, ignore=ignore_files)
    return


def get_config_params(yaml_fp:str, config_name:str) -> dict:

    with open(yaml_fp, 'r') as f: 
        all_params = yaml.load(f, Loader=SafeLoader)
    params = all_params[config_name]
    return  params


def get_trained_model_weight_paths(yaml_fp:str) -> dict:

    with open(yaml_fp, 'r') as f: 
        weights = yaml.load(f, Loader=SafeLoader)
    return  weights


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
    copy_tree('/Users/marco/Downloads/new_dataset/detection',
              '/Users/marco/Downloads/new_dataset/detection/temp',
              keep_format='.txt')