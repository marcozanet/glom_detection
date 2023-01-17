import os
import numpy as np
import yaml


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
    # create_pred_folder('/Users/marco/hubmap/tiles/dsrersa')
    edit_yaml(yaml_fp = 'dsohu', root = '/Users/marco/hubmap/data/' )