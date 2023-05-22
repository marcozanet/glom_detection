import os, shutil
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm
import json
import random
import matplotlib.pyplot as plt
import cv2


def show_image_labels(root:str, data_source:str, task:str, save = True):
    """ Shows K random images/labels. """

    replace_dir = lambda fp, to_dir, format: os.path.join(os.path.dirname(os.path.dirname(fp)), to_dir, os.path.basename(fp).split('.')[0] + f".{format}")
    labels = glob(os.path.join(root, task, 'tiles', '*', 'labels', '*.txt')) # /Users/marco/helical_tests/test_manager_detect_muw_sfog/detection/tiles/train
    assert len(labels)>0
    k=min(6, len(labels))

    # 1) Picking images:
    labels = random.sample(labels, k=k)
    pairs = [(replace_dir(fp, to_dir='images', format='png'), fp) for fp in labels]
    pairs = list(filter(lambda pair: (os.path.isfile(pair[0]) and os.path.isfile(pair[1])), pairs))
    # .log.info(f"Displaying {[os.path.basename(label) for label in labels]}")
    # 2) Show image/drawing rectangles as annotations:
    fig = plt.figure(figsize=(20, k//2*10))
    for i, (image_fp, label_fp) in enumerate(pairs):

        # read image
        image = cv2.imread(image_fp)
        W, H = image.shape[:2]
        # print((W, H))
        # .log.info(f"image shape: {W,H}")

        # read label
        with open(label_fp, 'r') as f:
            text = f.readlines()
            f.close()
        # draw rectangle for each glom/row:
        for row in text: 
            row = row.replace('/n', '')
            items = row.split(sep = ' ')
            class_n = int(float(items[0]))
            items = items[1:]
            x = [el for (j,el) in enumerate(items) if j%2 == 0]
            x = [np.int32(float(el)*W) for el in x]
            y = [el for (j,el) in enumerate(items) if j%2 != 0]
            y = [np.int32(float(el)*H) for el in y]
            vertices = list(zip(x,y)) 
            vertices = [list(pair) for pair in vertices]
            vertices = np.array(vertices, np.int32)
            vertices = vertices.reshape((-1,1,2))
            x0 = np.array(x).min()
            y0 = np.array(y).min()  
            if data_source == 'zaneta':    
                color = (0,255,0) if class_n == 0 else (255,0,0) 
            elif data_source == 'hubmap':
                # assert class_n == 0, f"Class to display is not 0, but hubmap should only contain class 0 objects."
                color = (0,255,0) if class_n == 0 else (255,0,0) 
            elif data_source == 'muw':
                color = (0,255,0) if class_n == 0 else (255,0,0) 
            image = cv2.fillPoly(image, pts = [vertices], color=color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'H' if class_n == 0 else 'U'
            image = cv2.putText(image, text, org = (x0,y0-H//50), color=color, thickness=3, fontFace=font, fontScale=1)

        # add subplot with image
        image = cv2.addWeighted(image, 0.4, cv2.imread(image_fp), 0.6, 1.0)
        plt.subplot(k//2,2,i+1)
        plt.title(f"Example tile.")
        plt.imshow(image)
        plt.tight_layout()
        plt.axis('off')
    
    plt.show()
    fig.savefig('tiled_examples.png')

    return



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

    # merge n_tiles.json files: 
    # read files:
    def _read_ntiles(_set_:str, dataset:str):
        ntiles_file = os.path.join(dataset, 'wsi', _set_, 'labels', 'n_tiles.json')
        if not os.path.isfile(ntiles_file):
            return None
        with open(ntiles_file, 'r') as f:
            data = json.load(f)
        return data
    # write files:
    def _write_merged_ntiles(_set_:str):
        ntiles_fp = os.path.join(dst_root, 'wsi', _set_, 'labels', 'n_tiles.json')
        with open(ntiles_fp, 'w') as f: 
            json.dump(merged_dict, fp=f)
        return
    for _set_ in sets:
        merged_dict = {}
        ntiles_dict1 = _read_ntiles(_set_=_set_, dataset=dataset1) # read dictionaries 
        ntiles_dict2 = _read_ntiles(_set_=_set_, dataset=dataset2)
        if ntiles_dict1 is not None:
            merged_dict.update(ntiles_dict1) # create merged one 
        if ntiles_dict2 is not None:
            merged_dict.update(ntiles_dict2)
        if len(merged_dict)>0:
            _write_merged_ntiles(_set_=_set_) # write merged one
    
    def merge_json_files(src:str, dst:str):
        """ Merges json files"""
        try:
            with open(src, 'r') as f: 
                data1 = json.load(f)
            with open(dst, 'r') as f: 
                data2 = json.load(f)
        except: 
            raise Exception(f"Couldn't read {dst} or {src}.")
        
        assert isinstance(data1, dict) and isinstance(data2, dict)
        json_merge = data2.update(data1)
        with open(dst, 'w') as f: 
            json.dump(json_merge, f)

        return json_merge

    # move data from dataset1:
    for src_fp, dst_fp in tqdm(src2dst1, desc='copying dataset1'):
        if safe_copy:
            if not os.path.isfile(dst_fp):
                shutil.copy(src=src_fp, dst=dst_fp)

    # move data from dataset2:
    for src_fp, dst_fp in tqdm(src2dst2, desc='copying dataset2'):
        if safe_copy:
            if not os.path.isfile(dst_fp) and 'n_tiles.json' not in dst_fp: #original n_tiles are not to be copied
                shutil.copy(src=src_fp, dst=dst_fp)
            elif os.path.isfile(dst_fp) and 'n_tiles.json' in src_fp and 'n_tiles.json' in dst_fp:
                merge_json_files(src=src_fp, dst=dst_fp)
                

    data12 = glob(os.path.join(dst_root, '*', '*', '*', '*'))
    len1, len2, len12 = len(data1), len(data2), len(data12)
    assert ((len1 + len2)-6)<len12<((len1 + len2)+6), f"Merged files are {len12}, but dataset1({len1}) + dataset2({len2}) = dataset12({len12})." # +-3 because n_tiles.json files are to be excluded.

    return

def switch_healthy_unhealthy_labels(dataset_root:str):

    assert os.path.isdir(dataset_root), f"'dataset_root':{dataset_root} is not a valid dirpath."
    assert 'tiles' in os.listdir(dataset_root), f"'dataset_root' should contain 'tiles'."

    tiles_labels = glob(os.path.join(dataset_root, 'tiles', '*', 'labels', '*.txt'))
    
    def _change_label(_fp:str):
        # read data
        with open(_fp,'r') as f:
            text = f.readlines()
        # loop throgu rows and objects and change 1st object = class
        new_text = []
        for row in text:
            objs = row.replace('\n', '').split(' ')
            class_n = int(float(objs[0]))
            assert class_n in [0,1,2,3,4,5,6,7,8,9], f"'class_n':{class_n} should be 0-9."
            new_class = '0' if class_n == 1 else '1'
            for obj in objs[1:]: # make new line
                new_row = new_class + ' ' + obj
            new_row += '\n'
            new_row = new_class + row[1:]
            assert len(new_row)>0
            new_text.append(new_row)
        # save new text
        with open(_fp, 'w') as f: 
            f.writelines(new_text)

        return

    for fp in tqdm(tiles_labels, desc='switching 0-1 labels'): 
        _change_label(_fp=fp)


    return


def test_switch_healthy_unhealthy_labels():
    dataset_root='/Users/marco/helical_datasets/muw_sfog/detection'

    switch_healthy_unhealthy_labels(dataset_root=dataset_root)


    return






def test_merge_datasets():

    dataset1 = '/Users/marco/helical_tests/test_merge_hubmap_sfog'
    dataset2 = '/Users/marco/Downloads/zaneta_files/detection' 
    dst_root = '/Users/marco/helical_tests/test_merge_hubmap_sfog_zaneta'
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
    # root='/Users/marco/Downloads/zaneta_files'
    # task='detection'
    # data_source='zaneta'
    # for _ in range(8):
    #     show_image_labels(root=root, data_source=data_source, task=task)