import os, sys, shutil
from glob import glob
from skimage import io
import numpy as np
import cv2
from tqdm import tqdm

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
        self.map_classes = map_classes
        self.resize = resize
        return
    

    def get_tot_gt_labels_from_dataset(self): 

        assert os.path.isdir(self.root_data), f"root_data:{self.root_data} is not a valid dirpath."
        tot_gt_labels = glob(os.path.join(self.root_data, 'tiles', "*", "labels", "*.txt"))
        assert len(tot_gt_labels)>0, f"No labels like {os.path.join(self.root_data, 'tiles', '*', 'labels', '*.txt')} found in dataset."
        return tot_gt_labels


    def get_image_shape(self): 
        """ Gets image dims"""
        img_dims: tuple
        assert os.path.isdir(root_data), f"root_data:{self.root_data} is not a valid dirpath."
        all_images = glob(os.path.join(self.root_data, 'tiles', "*", "images", "*.png"))
        assert len(all_images)>0, f"No labels like {os.path.join(self.root_data, 'tiles', '*', 'labels', '*.png')} found in dataset."
        image = io.imread(all_images[0])
        img_dims = image.shape

        return img_dims
    

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
            shutil.copy(src = old, dst = new)
        
        return
    

    def resize_images(self) -> None:
        """ Resizes images to match input size of the CNN (224, 224, 3) and saves them in the same folder."""

        # get all moved images: 
        tot_moved_images = glob(os.path.join(self.exp_data, 'crops_true_classes', '*', '*.jpg'))
        for img_fp in tqdm(tot_moved_images, 'resizing'): 
            image = cv2.imread(img_fp, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(224, 224))
            cv2.imwrite(img_fp, image)

        return


    def assign_all_labels(self) -> None:
        """ Assign a class to all crops and saving the mapping into the self.crop_class_dict"""
        
        tot_true_labels = {}
        for pred_lbl in self.tot_pred_labels: 
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
        gt_lbl = pred2gt_label(pred_lbl)

        if gt_lbl is None: 
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=0)
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            gt_classes = {crop_fp:None}
            return gt_classes
        
        with open(pred_lbl, 'r') as f: 
            pred_rows = f.readlines()

        with open(gt_lbl, 'r') as f: 
            gt_rows = f.readlines()

        def get_objs_from_row_txt_label(row:str):

            row = row.replace('\n', '')
            nums = row.split(' ')
            clss = int(nums[0])
            nums = [float(num) for num in nums[1:]]
            assert len(nums) == 4, f"there should be 4 objects apart from class"
            x_c, y_c, w, h = nums

            return clss, x_c, y_c, w, h

        gt_classes = {}
        for i, pred_row in enumerate(pred_rows):
            crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
            crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
            p_clss, p_xc, p_yc, p_w, p_h = get_objs_from_row_txt_label(pred_row)
            # now check if center of pred glom falls into any glom from true label (from gt_rows):
            matching_gloms = [] # gloms where predicted center for obj falls into
            for gt_row in gt_rows:
                g_clss, g_xc, g_yc, g_w, g_h = get_objs_from_row_txt_label(gt_row)
                min_x, max_x = max(g_xc - g_w/2, 0), min(g_xc + g_w/2, self.img_size[0])
                min_y, max_y = max(g_yc - g_h/2, 0), min(g_yc + g_h/2, self.img_size[1])
                if min_x<=p_xc<=max_x and min_y<=p_yc<=max_y:
                    matching_gloms.append((g_clss, g_xc, g_yc, g_w, g_h))
                
                # se ci sono vari match, devi assegnare la classe di quello col centro piu' vicino
                # tieni in un dizionario i match e poi guardi il piu' vicino 
            if len(matching_gloms) == 0: 
                crop_fn = lbl2cropfn(pred_lbl=pred_lbl, crop_n=i)
                crop_fp = cropfn_2_cropfp(crop_fn=crop_fn)
                gt_classes = {crop_fp:None}
                return gt_classes
            
            if len(matching_gloms) == 1:
                gt_class = matching_gloms[0][0]
            
            elif len(matching_gloms) > 1: 
                # compute dist between pred glom and the (many) matching gt_gloms
                min_dist = 99999999
                print("warning: multiple matching gloms for one pred glom in labeller not tested")
                for gt_glom in matching_gloms:
                    dist = eucl_dist(point1 = (p_xc, p_yc), point2= gt_glom[1:3])
                    min_dist = dist if dist < min_dist else min_dist
                    raise NotImplementedError()
                
            gt_classes.update({crop_fp:gt_class})

        return gt_classes


    def __call__(self) -> None:
        """ Assigns a label for all the crops, moves them in folds named after their true classes 
            and resizes images to match the CNN input. """

        self.assign_all_labels()
        self.move_assigned_crops()
        if self.resize: 
            self.resize_images()

        return



if __name__ == "__main__": 
    root_data = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection'
    exp_data = '/Users/marco/yolov5/runs/detect/exp30'
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} 
    resize = True
    labeller = CropLabeller(root_data=root_data, exp_data=exp_data, map_classes=map_classes, resize = resize)
    labeller()