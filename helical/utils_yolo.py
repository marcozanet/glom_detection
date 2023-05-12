import os, shutil
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm

def merge_datasets(dataset1: str, dataset2:str, dst_root:str, safe_copy:bool=True):
    """ Merges 2 datasets: train1+train1, val1+val2, test1+test2 """

    # create root structure
    image_types = ['tiles', 'wsi']
    sets = ['train', 'val', 'test']
    data_types = ['images', 'labels']
    for img_t in image_types: 
        for _set in sets: 
            for data_t in data_types:
                os.makedirs(os.path.join(dst_root, img_t, _set, data_t), exist_ok=True)
    
    # get data from dataset1 and change root dir to dst: 
    data1 = glob(os.path.join(dataset1, '*', '*', '*', '*'))
    data2 = glob(os.path.join(dataset2, '*', '*', '*', '*'))
    assert len(data1)> 0, f"'data1':{data1} is empty. No file like {os.path.join(dataset1, '*', '*', '*', '*')}"
    assert len(data2)> 0, f"'data2':{data2} is empty. No file like {os.path.join(dataset2, '*', '*', '*', '*')}"

    # compute new dst filepaths:
    change_fold = lambda fp, dataset: os.path.join(dst_root, os.path.relpath(fp, start=dataset))
    src2dst = lambda data, dataset:[(src_fp, change_fold(src_fp, dataset)) for src_fp in data ]
    src2dst1 = src2dst(data=data1, dataset=dataset1)
    src2dst2 = src2dst(data=data2, dataset=dataset2)

    # move data from dataset1:
    for src_fp, dst_fp in tqdm(src2dst1, desc='copying dataset1'):
        if safe_copy:
            if not os.path.isfile(dst_fp):
                shutil.copy(src=src_fp, dst=dst_fp)

    # move data from dataset1:
    for src_fp, dst_fp in tqdm(src2dst2, desc='copying dataset2'):
        if safe_copy:
            if not os.path.isfile(dst_fp):
                shutil.copy(src=src_fp, dst=dst_fp)
    data12 = glob(os.path.join(dst_root, '*', '*', '*', '*'))
    len1, len2, len12 = len(data1), len(data2), len(data12)
    assert ((len1 + len2)-3)<len12<((len1 + len2)+3), f"Merged files are {len12}, but dataset1({len1}) + dataset2({len2}) = dataset12({len12})." # +-3 because n_tiles.json files are to be excluded.

    return

def test_merge_datasets():

    dataset1 = '/Users/marco/helical_tests/test_manager_detect_muw_sfog/detection'
    dataset2 = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection'
    dst_root = '/Users/marco/helical_tests/test_merge_data'
    merge_datasets(dataset1=dataset1, dataset2=dataset2, dst_root=dst_root)

    return


def get_last_weights():
    """ Returns path to last trained model. """

    path_to_exps = '/Users/marco/yolov5/runs/train'
    files = os.listdir(path_to_exps)
    nums = [file.split('p')[1] for file in files if 'exp' in file]
    nums = np.array([int(file) for file in nums if len(file) > 0])
    last = str(nums.max())
    last = [file for file in files if last in file][0]
    last = os.path.join(path_to_exps, last, 'weights')
    weights = [os.path.join(last, file) for file in os.listdir(last) if 'best' in file][0]

    return weights


def get_last_detect():
    """ Returns path to last detected labels. """

    path_to_exps = '/Users/marco/yolov5/runs/detect'
    files = os.listdir(path_to_exps)
    nums = [file.split('p')[1] for file in files if 'exp' in file]
    nums = np.array([int(file) for file in nums if len(file) > 0])
    last = str(nums.max())
    last = [file for file in files if last in file][0]
    last = os.path.join(path_to_exps, last, 'labels')

    return last


def create_pred_folder(root: str):
    """ Creates folder for predictions. """

    subdirs = [fold for fold in os.listdir(root) if os.path.isdir(os.path.join(root,fold)) and 'preds_' in fold]
    if len(subdirs) == 0:
        os.makedirs(os.path.join(root, 'preds_0'))
        return
    nums = np.array([int(fold.split('_')[1]) for fold in subdirs])
    last_num = nums.max()
    new_num = str(last_num + 1)
    os.makedirs(os.path.join(root, f'preds_{new_num}'))

    return 


def move_detected_imgs(src_folder: str, dst_folder: str) -> None:
    """ Moves detected images from YOLO (png) into unet /images. """

    files = [file for file in os.listdir(src_folder) if 'png' in file and 'DS' not in file]
    for file in files:
        src = os.path.join(src_folder, file)
        dst = os.path.join(dst_folder, file)
        os.rename(src = src, dst = dst)

    return



def edit_yaml(root: str = False, test_folder: str = False ):
    """ Edits YAML data file from yolov5. """

    if isinstance(root, str) and test_folder is False:
        yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
        text = {'path':root, 'train': 'train/', 'val':'val/', 'test':'test/'}
        train, val, test = os.path.join(root, 'train'), os.path.join(root, 'val'), os.path.join(root, 'test')
        print(f"YOLO trains and test on: \n-{train} \n-{val} \n-{test}")
        with open(yaml_fp, 'w') as f:
            yaml.dump(data = text, stream=f)
    elif isinstance(test_folder, str) and root is False:
        yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
        print(f"YOLO test on: \n-{test_folder}")
        with open(yaml_fp, 'w') as f:
            yaml.dump(data = text, stream=f)
    else:
        raise TypeError(f"Params to edit_yaml should be path or False and either 'root' or 'test_folder' are to be specified.")

    return



if __name__ == '__main__':
    test_merge_datasets()