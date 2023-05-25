import os, sys, shutil
from glob import glob
from skimage import io
import numpy as np
import cv2
from tqdm import tqdm
from skimage.draw import rectangle
import albumentations as A
import random
root_data: str # from here ground truth label. Should be a dataset wsi/tiles -> train/val/test -> images/labels
exp_data: str # from here pred label

class CropLabeller(): 

    def __init__(self,
                 root_data:str,
                 exp_data:str,
                 map_classes:dict,
                 resize:bool = False) -> None:
        
        self.root_data = root_data   
        self.exp_data = exp_data
        self.img_size = self.get_image_shape()
        self.tot_gt_labels = self.get_tot_gt_labels_from_dataset()
        self.tot_pred_labels = glob(os.path.join(self.exp_data, 'labels', '*.txt'))
        self.tot_crops = glob(os.path.join(self.exp_data, 'crops', "*", '*.jpg'))
        self.map_classes = {v:k for v,k in map_classes.items() if k!='false_positives'} 
        self.resize = resize
        self.resize_shape = (224,224)

        return
    
    def _parse(self): 
        """ Parses arguments."""

        assert os.path.isdir(self.root_data), f"'root_data':{self.root_data} is not a valid dirpath."
        assert os.path.isdir(self.root_data), f"'exp_data':{self.exp_data} is not a valid dirpath."
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
        tot_gt_labels = glob(os.path.join(self.root_data, 'tiles', "*", "labels", "*.txt"))
        assert len(tot_gt_labels)>0, f"No labels like {os.path.join(self.root_data, 'tiles', '*', 'labels', '*.txt')} found in dataset."
        
        return tot_gt_labels


    def get_image_shape(self): 
        """ Gets image dims"""

        img_dims: tuple
        assert os.path.isdir(self.root_data), f"root_data:{self.root_data} is not a valid dirpath."
        all_images = glob(os.path.join(self.root_data, 'tiles', "*", "images", "*.png"))
        assert len(all_images)>0, f"No images like {os.path.join(self.root_data, 'tiles', '*', 'images', '*.png')} found in dataset."
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
        all_images = glob(os.path.join(self.exp_data, 'crops', "*", "*.jpg"))
        assert len(all_images)>0, f"No images like {os.path.join(self.exp_data, 'crops', '*', '*.jpg')} found in dataset."
        all_images = list(filter(self.is_image_to_copy, all_images))

        for img in tqdm(all_images, 'Padding crops'):
            self.resize_with_pad(image_fp=img, new_shape=new_shape, padding_color=padding_color)
        
        return
    

    def move_assigned_crops(self) -> None: 
        """ Copies crops into a folder which represent the true class for those crops. """

        # create folds:
        classes = list(self.map_classes.keys())
        os.makedirs(os.path.join(self.exp_data, 'crops_true_classes', 'false_positives'), exist_ok=True)
        for clss in classes:
            os.makedirs(os.path.join(self.exp_data, 'crops_true_classes', clss ), exist_ok=True)

        # move into correct folder:
        map_class2fold_map = {v:k for k,v in self.map_classes.items()}
        map_class2fold_map.update({None:'false_positives'})
        for crop_fp, crop_clss in self.crop_class_dict.items():
            old = crop_fp
            # fold_map
            new = os.path.join(self.exp_data, 'crops_true_classes', map_class2fold_map[crop_clss], os.path.basename(crop_fp))
            if self.is_image_to_copy(crop_fp=crop_fp):
                if not os.path.isfile(new):
                    shutil.copy(src = old, dst = new)
        
        return
    
    def is_image_to_copy(self, crop_fp:str)->bool: 
        """ Returns True if image is to copy. If e.g. image is a small crop, it is not to be copied. """
        
        is_to_copy = True
        crop = cv2.imread(crop_fp, cv2.COLOR_BGR2RGB)
        w, h = crop.shape[0], crop.shape[1]
        if w < h*0.7 or h < w*0.7:
            is_to_copy = False

        return is_to_copy
    

    def resize_images(self) -> None:
        """ Resizes images to match input size of the CNN (224, 224, 3) and saves them in the same folder."""

        # get all moved images: 
        tot_moved_images = glob(os.path.join(self.exp_data, 'crops_true_classes', '*', '*.jpg'))
        for img_fp in tqdm(tot_moved_images, 'Resizing crops'): 
            image = cv2.imread(img_fp, cv2.COLOR_BGR2RGB)
            try:
                image = cv2.resize(image, dsize=self.resize_shape)
            except:
                raise Exception(f"Couldn't resize {os.path.basename(img_fp)}")
                continue
            cv2.imwrite(img_fp, image)

        return


    def assign_all_labels(self) -> None:
        """ Assign a class to all crops and saving the mapping into the self.crop_class_dict"""

        if len(self.tot_pred_labels)==0:
            print(f"'tot_pred_labels' = glob({os.path.join(self.exp_data, 'labels', '*.txt')}) is empty.")
        tot_true_labels = {}
        for pred_lbl in tqdm(self.tot_pred_labels, desc='Labelling crops'): 
            gt_classes = self.assign_class(pred_lbl=pred_lbl)
            tot_true_labels.update(gt_classes)

        self.crop_class_dict = tot_true_labels

        return
    

    def assign_class(self, pred_lbl:str):
        """ Assign saved crop in inference to ground truth class. """

        # helper funcs 
        assert os.path.isfile(pred_lbl), f"pred_lbl:{pred_lbl} is not a valid filepath. "
        pred2gt_label = lambda pred_lbl: next((fp for fp in self.tot_gt_labels if os.path.basename(fp) == os.path.basename(pred_lbl)), None)
        eucl_dist = lambda point1, point2: np.sum(np.square(np.array(point1)-np.array(point2)))
        lbl2cropfn = lambda pred_lbl, crop_n: os.path.basename(pred_lbl).split('.txt')[0] + f"_crop{crop_n}"
        cropfn_2_cropfp = lambda crop_fn: next((fp for fp in self.tot_crops if crop_fn in fp), None)
        
        # look for matching true label from the original dataset
        gt_lbl = pred2gt_label(pred_lbl) # get true label
        if gt_lbl is None: # if doesn't exist, then it's a false pos
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=0)
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            assert os.path.isfile(crop_fp), f"crop_fp:{crop_fp} is not a valid filepath."
            gt_classes = {crop_fp:None}
            return gt_classes
        
        # otherwise look for each pred obj if its center falls into any of the true label objects:
        with open(pred_lbl, 'r') as f: # read pred label file
            pred_rows = f.readlines() # read true label file
        with open(gt_lbl, 'r') as f: 
            gt_rows = f.readlines()
        def get_objs_from_row_txt_label(row:str): # helper func
            row = row.replace('\n', '')
            nums = row.split(' ')
            clss = int(float(nums[0]))
            nums = [float(num) for num in nums[1:]]
            if len(nums) == 4:
                x_c, y_c, w, h = nums
            elif len(nums) == 8: 
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = nums
                x_c, y_c = x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2
                w, h = x_max-x_min, y_max-y_min
                assert all([el>=0 for el in [x_c, y_c, w, h]])
                assert x_c-w/2 == x_min, f"{x_c}-{w}/2 != {x_min}. Result is: {x_c-w/2}. "
            else:
                print(f"there should be 4 or 8 objects apart from class but are {len(nums)}")

            return clss, x_c, y_c, w, h
        
        
        # check if falls in any of the true objs
        gt_classes = {}
        for i, pred_row in enumerate(pred_rows): # for each pred obj
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            p_clss, p_xc, p_yc, p_w, p_h = get_objs_from_row_txt_label(pred_row)
            matching_gloms = [] 
            for gt_row in gt_rows: # for each true label obj
                g_clss, g_xc, g_yc, g_w, g_h = get_objs_from_row_txt_label(gt_row)
                min_x, max_x = max(g_xc - g_w/2, 0), min(g_xc + g_w/2, self.img_size[0])
                min_y, max_y = max(g_yc - g_h/2, 0), min(g_yc + g_h/2, self.img_size[1])
                if min_x<=p_xc<=max_x and min_y<=p_yc<=max_y: # look if pred center falls into true obj
                    if  g_w> 0.5 or g_h> 0.5: #NB IGNORE OBJECTS BIGGER THAN HALF THE TILE! (NO GLOMS LIKE THIS AT LEVEL 2 OF TILING)
                        continue 
                    matching_gloms.append((g_clss, g_xc, g_yc, g_w, g_h))
                    if len(matching_gloms)>1:
                        print(f"WARNING: glom in file {os.path.basename(pred_lbl)} has multiple matching true gloms. Assigning class of the closest.")
                        print(f"{min_x}<={p_xc}<={max_x} and {min_y}<={p_yc}<={max_y} \n matching gloms: {matching_gloms}")

            if len(matching_gloms) == 0: # if no pred obj does, maybe the model detected a part of an obj:
                pred_obj = {'p_clss':p_clss, 'p_xc':p_xc, 'p_yc':p_yc, 'p_w':p_w, 'p_h':p_h}
                for gt_row in gt_rows: # for each true label obj
                    g_clss, g_xc, g_yc, g_w, g_h = get_objs_from_row_txt_label(gt_row) 
                    gt_obj = {'g_clss':g_clss, 'g_xc':g_xc, 'g_yc':g_yc, 'g_w':g_w, 'g_h':g_h}
                    if self._is_predobj_part_of_trueobj(pred_obj=pred_obj, gt_obj=gt_obj): # check if is part of the true obj
                        matching_gloms.append((g_clss, g_xc, g_yc, g_w, g_h))
                else:
                    crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
                    crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
                    gt_classes = {crop_fp:None}
                    return gt_classes
            
            if len(matching_gloms) == 1:
                gt_class = matching_gloms[0][0]
            
            elif len(matching_gloms) > 1: # if multiple matches for same pred obj:
                min_dist = 99999999
                for j, gt_glom in enumerate(matching_gloms):
                    dist = eucl_dist(point1 = (p_xc, p_yc), point2= gt_glom[1:3])
                    min_dist = dist if dist < min_dist else min_dist
                    min_class = matching_gloms[j][0]
                gt_class = min_class
                
            gt_classes.update({crop_fp:gt_class})
            
        return gt_classes
    
    def balance_dataset(self):

        # get images
        classes_folds = [ os.path.join(self.exp_data, 'crops_true_classes', fold) for fold in os.listdir(os.path.join(self.exp_data, 'crops_true_classes'))]
        classes_folds = [ fold for fold in classes_folds if os.path.isdir(fold)]
        assert len(classes_folds) > 0
        classes_n_images = {clss_fp: len(os.listdir(clss_fp)) for clss_fp in classes_folds}
        max_n_imgs = max(classes_n_images.values())

        # albumentations funcs
        transform = A.Compose([
            A.ToGray(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2)])
        change_name_augm = lambda fp, i: os.path.join(os.path.dirname(fp), f"Augm_{i}_{os.path.basename(fp)}")
        
        
        for clss_fp, n_imgs in classes_n_images.items():

            # skip max class, augment others
            if n_imgs == max_n_imgs:
                continue 

            # get images
            clss_images = glob(os.path.join(clss_fp, '*.jpg'))
            clss_images = [img for img in clss_images if 'Augm' not in img]

            assert len(clss_images)>0, f"No images like {os.path.join(clss_fp, '*.jpg')}"
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
        self.pad_all_images(new_shape=self.resize_shape)
        self.assign_all_labels()
        self.move_assigned_crops()
        if self.resize: 
            self.resize_images()
        
        self.balance_dataset()

        return



if __name__ == "__main__": 
    root_data = '/Users/marco/helical_tests/test_merge_muw_zaneta'
    exp_data = '/Users/marco/helical_tests/test_cnn_trainer2/exp40'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1} 
    resize = True
    labeller = CropLabeller(root_data=root_data, exp_data=exp_data, 
                            map_classes=map_classes, resize = resize)
    labeller()