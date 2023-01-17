import yaml
import time
import os
import numpy as np
import json


def edit_yaml(task:str,
              root:str = False,
              system:str = 'windows',
              classes:dict = {0: 'healthy', 1: 'unhealthy'}) -> None:
    """ Edits YAML data file from yolov5. """

    assert task in ['segmentation', 'detection'], f"'task' should be either 'segmentation' or 'detection'."

    if system == 'mac':
        yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
        text = {'path':root, 'train': f'{task}/train/images/', 'val':f'{task}/val/images/', 'test':f'{task}/test/images/', 'names':classes}
    elif system == 'windows':
        yaml_fp = r'C:\marco\yolov5\data\hubmap.yaml'
        text = {'path':root, 'train': f"{task}\\train\\images", 'val': f"{task}\\val\\images", 'test':f'{task}\\test\\images','names':classes}
    with open(yaml_fp, 'w') as f:
        yaml.dump(data = text, stream=f)

    return


def write_YOLO_txt(add_params: dict, root_exps: str = '/Users/marco/yolov5/runs/train') -> None:
    """ Given the additional params, it writes a filetext and saves them into the last exp dir in root_exps. 
        add_params = dictionary to be written into the json file.
        root_exps = root to the YOLO train experiments. """
    
    # 1) get last YOLO train exp dir
    exp_dirs = sorted([os.path.join(root_exps, dir) for dir in os.listdir(root_exps) if os.path.isdir(os.path.join(root_exps, dir)) and 'exp' in dir])
    last_dir = exp_dirs[-1]
    print(last_dir)
    # 2) write text 
    txt_fp = os.path.join(last_dir, f"other_info.json")
    add_params = json.dumps(add_params, indent=4, sort_keys=True, default=str)
    with open(txt_fp, 'w') as f:
        json.dump(add_params, f)

    return


def test_write_YOLO_txt():

    add_params = {'data': 'muw', 'classes': {0: 'healthy', 1: 'unhealthy'}}
    write_YOLO_txt(add_params)

    return






if __name__ == '__main__':

    test_write_YOLO_txt()