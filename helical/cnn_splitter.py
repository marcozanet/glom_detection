
import os, shutil
from glob import glob
from sklearn import model_selection as ms
from tqdm import tqdm
import cv2

class CNNDataSplitter(): 


    def __init__(self, 
                 src_folds:list,
                 yolo_root:str,
                 dst_root: str,
                 map_classes: dict,
                 treat_as_single_class:bool=False) -> None:
        """ Prepares data for CNN. Crops are divided by true class, resized and split in datasets."""

        self.src_folds = src_folds # exp folds where images for train, val, test can be retrieved
        self.map_classes = map_classes
        self.treat_as_single_class = treat_as_single_class
        self.class_folds = self._get_class_folds()
        self.tot_images = self._get_all_files()
        self.dst_root = dst_root
        self.yolo_root = yolo_root
        self.map_classes = {v:k for v,k in map_classes.items() if k!='false_positives'} 


        return
    

    def _get_class_folds(self):
        """ Get class folds """

        class_folds = list(self.map_classes.keys())
        class_folds.append('false_positives')
        for exp_fold in self.src_folds: 
            found_folds = os.listdir(os.path.join(exp_fold, 'crops_true_classes'))
            found_folds = [dir for dir in found_folds if "DS" not in dir]
            assert set(class_folds) == set(found_folds), f"Class folds should be {class_folds}, but found class folds are {found_folds}"

        return list(set(class_folds))
    
    
    # def _resize_images(self) -> None:
    #     """ Resizes images to match input size of the CNN (224, 224, 3) and saves them in the same folder."""

    #     # get all moved images: 
    #     tot_moved_images = glob(os.path.join(self.dst_root, '*', '*', '*.jpg'))
    #     tot_moved_images = [file for file in tot_moved_images if "DS" not in file]
    #     assert len(tot_moved_images)>0, f"No image like {os.path.join(self.dst_root, '*', '*', '*.jpg')} found in dst_root."

    #     for img_fp in tqdm(tot_moved_images, 'resizing'): 
    #         # print(img_fp)
    #         image = cv2.imread(img_fp, cv2.COLOR_BGR2RGB)
    #         image = cv2.resize(image, dsize=(224, 224))
    #         cv2.imwrite(img_fp, image)

    #     return
    

    def _get_all_files(self):
        """ Gets tile names, slide names etc. """

        tot_images = []
        for exp_fold in self.src_folds:
            assert os.path.isdir(exp_fold), f"exp_fold:{exp_fold} is not a valid dirpath."
            images = glob(os.path.join(exp_fold, 'crops_true_classes', '*', '*.jpg'))
            assert len(images)>0, f"No image like {os.path.join(exp_fold, 'crops_true_classes', '*', '*.jpg')} found."
            tot_images.extend(images)

        return tot_images
    

    def _treat_as_single_class(self):
        """ Merges all classes other than 'false_positives' altogether in 'item' 
            resulting classes used by the model are 'false_positives', 'item'. """
        
        all_images = glob(os.path.join(self.dst_root, '*', '*', '*.jpg'))
        assert len(all_images) > 0, f"No image like {os.path.join(self.dst_root, '*', '*', '*.jpg')} "

        # create 'item' folds:
        sets = ['train', 'val', 'test']
        class_folds = ['false_positives', 'item']
        for dataset in sets: 
            for clss in class_folds: 
                os.makedirs(os.path.join(self.dst_root, dataset, clss), exist_ok=True)

        # move all images to 'item' class
        change_class = lambda fp: os.path.join(self.dst_root, os.path.split(os.path.dirname(os.path.dirname(fp)))[1], 'item', os.path.basename(fp)) 
        for fp in tqdm(all_images, desc=f"merging classes"): # to not be done on 'false_positives'
            # skip false_positives:
            if os.path.split(os.path.dirname(fp))[1] == 'false_positives':
                continue
            # move others:
            src = fp
            dst = change_class(fp)
            if not os.path.isfile(dst):
                shutil.move(src=src, dst=dst)
        
        # delete other class folds
        del_classes = [_cls for _cls in self.class_folds if _cls not in ['false_positives', 'item']]
        for dataset in sets: 
            for clss in del_classes: 
                del_fp = os.path.join(self.dst_root, dataset, clss)
                if os.path.isdir(del_fp):
                    shutil.rmtree(del_fp)

        return
    
    
    def _make_dataset(self) -> None: 
        """ Creates dataset folds. """
        
        os.makedirs(self.dst_root, exist_ok=True)
        sets = ['train', 'val', 'test']
        self.traindir = os.path.join(self.dst_root, 'train')
        self.valdir = os.path.join(self.dst_root, 'val')
        self.testdir = os.path.join(self.dst_root, 'test')

        class_folds = self.class_folds
        for dataset in sets: 
            if os.path.isdir(os.path.join(self.dst_root, dataset)):
                print(f"Found dataset: deleting {dataset}")
                shutil.rmtree(os.path.join(self.dst_root, dataset))
            for clss in class_folds: 
                os.makedirs(os.path.join(self.dst_root, dataset, clss), exist_ok=True)

        return
    
    
    def _split_move_images(self, test_size:float=0.2):
        """ Splits crops from exp folder into train, val, test and moves them in the new folds. """

        # map_classint2classname = {v:k for k,v in self.map_classes.items()}
        tot_map_classes = self.map_classes
        tot_map_classes.update({'false_positives':999})
        x = self.tot_images
        y = [tot_map_classes[os.path.split(os.path.dirname(img))[1]] for img in self.tot_images]
        train_imgs, test_imgs, _, _ = ms.train_test_split(x, y, test_size=test_size, stratify=y) # stratify=y)

        # move train images:
        for fp in tqdm(train_imgs, 'filling train:'): 
            src = fp
            dst = os.path.join(self.traindir, os.path.split(os.path.dirname(src))[1], os.path.basename(src))
            shutil.copy(src=src, dst=dst)
        # move test images:
        for fp in tqdm(test_imgs, 'filling test:'): 
            src = fp
            dst = os.path.join(self.testdir, os.path.split(os.path.dirname(src))[1], os.path.basename(src))
            shutil.copy(src=src, dst=dst)


        return
    
    
    def __call__(self):
        # self._get_all_files()
        self._make_dataset()
        self._split_move_images()
        # self._resize_images()
        if self.treat_as_single_class:
            self._treat_as_single_class()
        
        return
    

if __name__ == "__main__": 
    # root_data = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection'
    # exp_data = '/Users/marco/yolov5/runs/detect/exp30'
    src_folds = ['/Users/marco/yolov5/runs/detect/exp30']
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} 
    dst_root = '/Users/marco/helical_tests/test_cnn_processor'
    resize = True
    # task = 'detection'
    cnn_processor = CNNDataSplitter(src_folds=src_folds, map_classes=map_classes, dst_root=dst_root, resize=resize)
    cnn_processor()