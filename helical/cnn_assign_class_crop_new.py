import os, sys, shutil
from glob import glob
from skimage import io
import numpy as np
import cv2
from tqdm import tqdm
from skimage.draw import rectangle
import albumentations as A
import random
from utils import get_config_params
from loggers import get_logger
from configurator import Configurator

class CropLabeller(Configurator): 

    def __init__(self, config_yaml_fp:str, exp_fold:str, skipped_crops:list)-> None:

        super().__init__()
        self.params = get_config_params(config_yaml_fp, 'cnn_trainer')
        self.root_data = os.path.join(self.params["root_data"] )
        self.exp_fold = exp_fold 
        self.tot_pred_labels = glob(os.path.join(self.exp_fold, 'labels', '*.txt'))
        self.tot_crops = glob(os.path.join(self.exp_fold, 'crops', "*", '*.jpg'))
        self.map_classes = {v:k for v,k in self.params['map_classes'].items() if k!='false_positives'} 
        # self.resize = self.params['resize_crops']
        self.yolo_task = self.params['yolo_task']
        self.resize_shape = tuple(self.params['resize_shape']) if self.params['resize_shape'] is not None else (224,224)
        # print(f"Resize_shape':{self.resize_shape}")
        self.skipped_crops = skipped_crops
        # raise NotImplementedError()
        self.img_size = self.get_image_shape()
        self.tot_gt_labels = self.get_tot_gt_labels_from_dataset()
        return
    

    def _parse(self): 
        """ Parses arguments."""

        assert os.path.isdir(self.root_data), f"'root_data':{self.root_data} is not a valid dirpath."
        assert os.path.isdir(self.root_data), f"'exp_data':{self.exp_fold} is not a valid dirpath."
        assert type(self.map_classes) == dict, f"'map_classes':{self.map_classes} should be a dict, but is type {type(self.map_classes)}."

        return
    

    def _is_predobj_part_of_trueobj(self, pred_obj:dict, gt_obj:dict, iou_thr:float=0.25):
        """ Parts of objects are also to be counted as true pos -> x = intersection/pred ~= 1
            -> 0.8 < x < 1. """
        
        # get all vertices:
        _, p_xc, p_yc, p_w, p_h = pred_obj['p_clss'], pred_obj['p_xc'], pred_obj['p_yc'], pred_obj['p_w'], pred_obj['p_h']
        _, g_xc, g_yc, g_w, g_h = gt_obj['g_clss'], gt_obj['g_xc'], gt_obj['g_yc'], gt_obj['g_w'], gt_obj['g_h']
        p_x0, p_x1 = max((p_xc - p_w/2), 0), min((p_xc + p_w/2), self.img_size[0])
        p_y0, p_y1 = max((p_yc - p_h/2), 0), min((p_yc + p_h/2), self.img_size[1])
        g_x0, g_x1 = max((g_xc - g_w/2), 0), min((g_xc + g_w/2), self.img_size[0])
        g_y0, g_y1 = max((g_yc - g_h/2), 0), min((g_yc + g_h/2), self.img_size[1])
        
        # back to original pixel scale: 
        p_x0, p_x1, g_x0, g_x1 = [int(val*self.img_size[0]) for val in [p_x0, p_x1, g_x0, g_x1]]
        p_y0, p_y1, g_y0, g_y1 = [int(val*self.img_size[1]) for val in [p_y0, p_y1, g_y0, g_y1]]

        # draw masks
        pred_img, true_img = np.zeros(self.img_size[:2], dtype=np.uint8), np.zeros(self.img_size[:2], dtype=np.uint8)
        pred_mask = rectangle(start=(p_x0, p_y0), end=(p_x1, p_y1), shape = pred_img.shape)
        gt_mask = rectangle(start=(g_x0, g_y0), end=(g_x1, g_y1), shape = true_img.shape)
        pred_img[pred_mask] = 1
        true_img[gt_mask] = 1

        # compute intersection/pred obj (1 if whole pred obj is inside true obj, 0 if no intersection)
        pred_intersection = float((pred_img * true_img).sum()) /  float(pred_img.sum())
        assert pred_intersection <=1, f"pred_intersection:{pred_intersection}, but shouldn't be >1 by definition."
        matching = True if iou_thr<=pred_intersection<=1 else False

        return matching
    

    def get_tot_gt_labels_from_dataset(self): 
        """ Retrieves true labels from the original data folder. """

        assert os.path.isdir(self.root_data), f"root_data:{self.root_data} is not a valid dirpath."
        path_like = os.path.join(self.root_data, self.yolo_task, 'tiles', "*", "labels", "*.txt")
        tot_gt_labels = glob(path_like)
        assert len(tot_gt_labels)>0, f"No labels like {path_like} found in dataset."
        
        return tot_gt_labels


    def get_image_shape(self): 
        """ Gets image dims"""

        img_dims: tuple
        assert os.path.isdir(self.root_data), f"root_data:{self.root_data} is not a valid dirpath."
        path_like = os.path.join(self.root_data, self.yolo_task, 'tiles', "*", "images", "*.png")
        all_images = glob(path_like)
        assert len(all_images)>0, f"No images like {path_like} found in dataset."
        image = io.imread(all_images[0])
        img_dims = image.shape

        return img_dims
    
    
    def resize_with_pad(self, image_fp: np.array, new_shape: tuple,
                        padding_color: tuple = (255, 255, 255)) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        
        image = cv2.imread(image_fp, cv2.COLOR_BGR2RGB)
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        cv2.imwrite(filename=image_fp, img=image)

        return 
    
    
    def pad_all_images(self, new_shape: tuple, padding_color: tuple = (255, 255, 255)) -> np.array:
        """ Pads all images into a new shape. """

        assert os.path.isdir(self.root_data), f"root_data:{self.root_data} is not a valid dirpath."
        path_like = os.path.join(self.exp_fold, 'crops', "*", "*.jpg")
        all_images = glob(path_like)
        assert len(all_images)>0, f"No images like {path_like} found in dataset."
        all_images = list(filter(self.is_image_to_copy, all_images))

        for img in tqdm(all_images, 'Padding crops'):
            self.resize_with_pad(image_fp=img, new_shape=new_shape, padding_color=padding_color)
        
        return
    

    def move_assigned_crops(self) -> None: 
        """ Copies crops into a folder which represent the true class for those crops. """
        func_n = self.move_assigned_crops.__name__

        # create folds:
        classes = list(self.map_classes.keys())
        os.makedirs(os.path.join(self.exp_fold, 'crops_true_classes', 'false_positives'), exist_ok=True)
        for clss in classes:
            os.makedirs(os.path.join(self.exp_fold, 'crops_true_classes', clss ), exist_ok=True)

        # move into correct folder:
        map_class2fold_map = {v:k for k,v in self.map_classes.items()}
        map_class2fold_map.update({None:'false_positives'}) # adding false positive class for base class = 0 e.g. Glomerulus (wo classification)
        for crop_fp, crop_clss in self.crop_class_dict.items():
            try:
                map_class2fold_map[crop_clss]
            except:
                msg = f"Crop seems to be class {crop_clss}, which is not one of 'map_classes' values: {self.map_classes}"
                self.format_msg(msg=msg, func_n=func_n, type='error')
                raise Exception(msg)
            old = crop_fp

            if crop_fp is None: continue
            assert os.path.isfile(crop_fp), self.assert_log(f"'crop_fp':{crop_fp} doesn't exist .", func_n=func_n)
            # fold_map
            new = os.path.join(self.exp_fold, 'crops_true_classes', map_class2fold_map[crop_clss], os.path.basename(crop_fp))
            if self.is_image_to_copy(crop_fp=crop_fp):
                if not os.path.isfile(new):
                    shutil.copy(src = old, dst = new)
        
        return
    

    def is_image_to_copy(self, crop_fp:str)->bool: 
        """ Returns True if image is to copy. If e.g. image is a small crop, it is not to be copied. """
        
        is_to_copy = True
        crop = cv2.imread(crop_fp, cv2.COLOR_BGR2RGB)
        w, h = crop.shape[0], crop.shape[1]
        if w < h*0.6 or h < w*0.6:
            is_to_copy = False

        return is_to_copy
    

    def resize_images(self) -> None:
        """ Resizes images to match input size of the CNN (224, 224, 3) and saves them in the same folder."""

        # get all moved images: 
        tot_moved_images = glob(os.path.join(self.exp_fold, 'crops_true_classes', '*', '*.jpg'))
        for img_fp in tqdm(tot_moved_images, 'Resizing crops'): 
            image = cv2.imread(img_fp, cv2.COLOR_BGR2RGB)
            try:
                image = cv2.resize(image, dsize=self.resize_shape)
            except:
                raise Exception(f"Couldn't resize {os.path.basename(img_fp)}")
            cv2.imwrite(img_fp, image)

        return


    def assign_all_labels(self) -> None:
        """ Assign a class to all crops and saving the mapping into the self.crop_class_dict"""

        if len(self.tot_pred_labels)==0:
            print(f"'tot_pred_labels' = glob({os.path.join(self.exp_fold, 'labels', '*.txt')}) is empty.")
        tot_true_labels = {}
        for pred_lbl in tqdm(self.tot_pred_labels, desc='Labelling crops'): 
            gt_classes = self.assign_class(pred_lbl=pred_lbl)
            if gt_classes is not None: tot_true_labels.update(gt_classes)

        self.crop_class_dict = tot_true_labels

        return


    def _get_objs_from_row_txt_label(self, row:str): # helper func
        row = row.replace('\n', '')
        nums = row.split(' ')
        clss = int(float(nums[0]))
        nums = [float(num) for num in nums[1:]]
        # detection case:
        if len(nums) == 4:
            x_c, y_c, w, h = nums
        # segmentation case:
        elif len(nums) == 8: 
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = nums
            x_c, y_c = x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2
            w, h = x_max-x_min, y_max-y_min
            assert all([el>=0 for el in [x_c, y_c, w, h]])
            assert x_c-w/2 == x_min, f"{x_c}-{w}/2 != {x_min}. Result is: {x_c-w/2}. "
        else:
            print(f"there should be 4 or 8 objects apart from class but are {len(nums)}")

        return clss, x_c, y_c, w, h    


    def assign_class(self, pred_lbl:str):
        """ Assign saved crop in inference to ground truth class. """

        error = True if 'BH13_PAS_sample0_11_5_crop2' in pred_lbl else False
        # print('a')
        # helper funcs 
        assert os.path.isfile(pred_lbl), f"pred_lbl:{pred_lbl} is not a valid filepath. "
        pred2gt_label = lambda pred_lbl: next((fp for fp in self.tot_gt_labels if os.path.basename(fp) == os.path.basename(pred_lbl)), None)
        eucl_dist = lambda point1, point2: np.sum(np.square(np.array(point1)-np.array(point2)))
        lbl2cropfn = lambda pred_lbl, crop_n: os.path.basename(pred_lbl).split('.txt')[0] + f"_crop{crop_n}"
        cropfn_2_cropfp = lambda crop_fn: next((fp for fp in self.tot_crops if crop_fn in fp), None)
        # print('b')
        
        # look for matching true label from the original dataset
        gt_lbl = pred2gt_label(pred_lbl) # get true label
        # print('b1')
        if error is True: self.log.error(gt_lbl)
        # print('b2')
        if error is True: self.log.error(pred_lbl)

        if gt_lbl is None: # if doesn't exist, then it's a false pos
            # print('b4')
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=0)
            # print('b5')
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            # print('b6')
            # print(crop_fp)
            if crop_fp is None: return
            assert os.path.isfile(crop_fp), f"crop_fp:{crop_fp} is not a valid filepath."
            gt_classes = {crop_fp:None}
            return gt_classes
        
        # print('c')
        # otherwise look for each pred obj if its center falls into any of the true label objects:
        with open(pred_lbl, 'r') as f: # read pred label file
            pred_rows = f.readlines() # read true label file
        with open(gt_lbl, 'r') as f: 
            gt_rows = f.readlines()
        
        # check if falls in any of the true objs
        gt_classes = {}
        for i, pred_row in enumerate(pred_rows): # for each pred obj
            # print('d')
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
            # print('e')
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            # print('f')
            p_clss, p_xc, p_yc, p_w, p_h = self._get_objs_from_row_txt_label(pred_row)
            if error is True: self.log.error(f"Pred obj: {p_clss, p_xc, p_yc, p_w, p_h}")

            matching_gloms = [] 
            zs= []
            # print('g')
            for z, gt_row in enumerate(gt_rows): # for each true label obj
                g_clss, g_xc, g_yc, g_w, g_h = self._get_objs_from_row_txt_label(gt_row)
                if error is True: self.log.error(f"GT obj: {g_clss, g_xc, g_yc, g_w, g_h}")
                min_x, max_x = max(g_xc - g_w/2, 0), min(g_xc + g_w/2, self.img_size[0])
                min_y, max_y = max(g_yc - g_h/2, 0), min(g_yc + g_h/2, self.img_size[1])
                if min_x<=p_xc<=max_x and min_y<=p_yc<=max_y: # look if pred center falls into true obj
                    matching_gloms.append((g_clss, g_xc, g_yc, g_w, g_h))

            # self.log.info((g_clss, g_xc, g_yc, g_w, g_h))
            # if doesn't -> it's a false pos
            # print('h')
            if len(matching_gloms) == 0: 
                crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
                crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
                print(f"FP found!")
                gt_classes.update({crop_fp:None})
            
            # print('i')
            # if it does -> it's an obj of the base class 0 (e.g. Glomerulus)
            if len(matching_gloms) == 1:
                if self.params['mutually_exclusive_classes'] is False: 
                    gt_class = self._refine_assignment_wclassification(gt_lbl=gt_lbl, matching_glom=matching_gloms[0])
                    # self.log.info(gt_class)
                else:
                    gt_class = matching_gloms[0][0]
                gt_classes.update({crop_fp:gt_class})
            
            elif len(matching_gloms) > 1: # if multiple matches for same pred obj:
                min_dist = 99999999
                for j, gt_glom in enumerate(matching_gloms):
                    dist = eucl_dist(point1 = (p_xc, p_yc), point2= gt_glom[1:3])
                    min_dist = dist if dist < min_dist else min_dist
                    min_class = matching_gloms[j][0]
                gt_class = min_class
                gt_classes.update({crop_fp:gt_class})

        return gt_classes
    
    
    def _refine_assignment_wclassification(self, gt_lbl:str, matching_glom:tuple):
        """ 'assign_class' assigns a basic class without classification -> either FP or Glomerulous. 
            This function refines the assignement by checking on the saved 'temp' tree with classification labels too.
            Output: FP, Healthy, Fibrous, CellularCrescent..."""
        func_n = self._refine_assignment_wclassification.__name__
        # self.log.info(f"Called refine assignment ")
        
        temp_tree = os.path.join(self.root_data, self.params['yolo_task'], 'temp')
        assert os.path.isdir(temp_tree), self.assert_log(f"'temp' tree doesn't exist at {temp_tree}", func_n=func_n)
        
        # get corresponding true label with also classification from the temp tree:
        # gt_lbl_temp: '/Users/marco/Downloads/new_dataset/detection/temp/tiles/train/labels/BH13_PAS_sample0_0_3.txt'
        # gt_lbl: '/Users/marco/Downloads/new_dataset/detection/tiles/train/labels/BH13_PAS_sample0_0_3.txt'

        assert self.root_data in gt_lbl, self.assert_log(f"{self.root_data} not in {gt_lbl}", func_n=func_n)


        gt_lbl_temp_path_like = os.path.join(temp_tree, 'tiles', '*', 'labels', os.path.basename(gt_lbl))
        temp_matches = glob(gt_lbl_temp_path_like)
        assert len(temp_matches)==1, self.assert_log(f"'temp_matches' has length: {temp_matches}.", func_n=func_n)
        gt_lbl_temp = temp_matches[0]

        assert os.path.isfile(gt_lbl_temp), self.assert_log(f"'gt_lbl_temp':{gt_lbl_temp} is not a valid filepath.", func_n=func_n)

        match_clss, match_xc, match_yc, match_w, match_h = matching_glom
        min_x, max_x = max(match_xc - match_w/2, 0), min(match_xc + match_w/2, self.img_size[0])
        min_y, max_y = max(match_yc - match_h/2, 0), min(match_yc + match_h/2, self.img_size[1])

        with open(gt_lbl_temp, 'r') as f: 
            gt_temp_rows = f.readlines()

        # (pred_lbl), gt_lbl, gt_lbl_temp
        # self.log.info(f"Label temp:{gt_lbl_temp}")
        found = []
        for z, gt_temp_row in enumerate(gt_temp_rows): # for each true label obj
            g_wc_clss, g_wc_xc, g_wc_yc, g_wc_w, g_wc_h = self._get_objs_from_row_txt_label(gt_temp_row)
            # self.log.info(f"Matching glom: {matching_glom}")
            # self.log.info(f"Comparing with {g_wc_clss, g_wc_xc, g_wc_yc, g_wc_w, g_wc_h}")
            if (g_wc_clss, g_wc_xc, g_wc_yc, g_wc_w, g_wc_h) == matching_glom:
                # self.log.info(f"Just self glom. skipping")
                continue
            if (min_x<=g_wc_xc<=max_x) and (min_y<=g_wc_yc<=max_y): # a class obj has been found inside glom
                # self.log.info(f"Found: Condition respected: ({min_x}<={g_wc_xc}<={max_x}) and ({min_y}<={g_wc_yc}<={max_y})  ")
                found.append((g_wc_clss, g_wc_xc, g_wc_yc, g_wc_w, g_wc_h))
            # else:
            #     self.log.info(f"Condition not respected: ({min_x}<={g_wc_xc}<={max_x}) and ({min_y}<={g_wc_yc}<={max_y}) ")

        if len(found)==0:
            return 0
        elif len(found)==1:
            return g_wc_clss
        else:
            unique_classes = list(set([values[0] for values in found]))
            if len(unique_classes) == 1:
                # self.log.info(f"case unique classes. Class: {g_wc_clss}")
                return found[0][0]
            else:
                self.log.error(f"Inside:{matching_glom} were found these objects: {found} with different classes: {unique_classes}. Which one prevails?")
                raise NotImplementedError()

    
    def balance_dataset(self):

        # get images
        classes_folds = [ os.path.join(self.exp_fold, 'crops_true_classes', fold) for fold in os.listdir(os.path.join(self.exp_fold, 'crops_true_classes'))]
        classes_folds = [ fold for fold in classes_folds if os.path.isdir(fold)]
        assert len(classes_folds) > 0
        classes_n_images = {clss_fp: len(os.listdir(clss_fp)) for clss_fp in classes_folds}
        max_n_imgs = max(classes_n_images.values())
        # albumentations funcs
        transform = A.Compose([
            A.OneOf([A.RandomContrast(p=0.3)]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)])
        change_name_augm = lambda fp, i: os.path.join(os.path.dirname(fp), f"Augm_{i}_{os.path.basename(fp)}")
        
        
        for clss_fp, n_imgs in classes_n_images.items():

            # skip max class, augment others
            if n_imgs == max_n_imgs:
                continue 

            # get images
            clss_images = glob(os.path.join(clss_fp, '*.jpg'))
            clss_images = [img for img in clss_images if 'Augm' not in img]

            if len(clss_images)==0: self.log.warning( f"No images like {os.path.join(clss_fp, '*.jpg')}. Skipping. DATASET WILL BE UNBALANCED.")
            if len(clss_images)==0: continue

            n_imgs_to_make = max_n_imgs - n_imgs

            print(f"Images to create: {n_imgs_to_make}")

            # create new images:
            for i in tqdm(range(n_imgs_to_make), desc=f'Augmenting {os.path.basename(clss_fp)}'): 
                img_fp = random.choice(clss_images)
                img = cv2.imread(img_fp, cv2.COLOR_BGR2RGB)
                transformed_img = transform(image=img)['image']
                write_fp = change_name_augm(img_fp, i)
                if not os.path.isfile(write_fp):
                    cv2.imwrite(write_fp, transformed_img)
            
            n_clss_imgs = len(glob(os.path.join(clss_fp, '*.jpg')))
            assert n_clss_imgs == max_n_imgs
            

        return


    def __call__(self) -> None:
        """ Assigns a label for all the crops, moves them in folds named after their true classes 
            and resizes images to match the CNN input. """
        
        self._parse()
        # self.pad_all_images(new_shape=self.resize_shape)
        self.assign_all_labels()
        self.move_assigned_crops()
        self.resize_images()
        self.balance_dataset()

        return



if __name__ == "__main__": 
    config_yaml_fp = '/Users/marco/yolo/code/helical/tcd_config_training.yaml'
    labeller = CropLabeller(config_yaml_fp=config_yaml_fp, exp_fold='/Users/marco/yolov5/runs/detect/exp83')
    labeller()