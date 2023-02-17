import os
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import List
from skimage import io
import seaborn as sns
import matplotlib.pyplot as plt
from loggers import get_logger



class Cropper():
    """ Crops images out of the YOLO. """

    def __init__(self, 
                root:str, 
                save_folder:str, 
                image_shape:tuple, 
                crop_shape:tuple = None,
                percentile:int = 95,
                plot_boxes:bool = False,
                skip_cropped_objects: bool = True) -> None:
        
        self.log = get_logger()
        
        # assertions
        assert isinstance(image_shape, tuple) and image_shape[0] == image_shape[1], f"'image_shape' should be a tuple of equal int values."
        assert os.path.isdir(root), ValueError(f"'root_dir': {root} is not a valid dirpath. ")
        assert os.path.isdir(save_folder), ValueError(f"'save_folder': {save_folder} is not a valid dirpath. ")
        assert (isinstance(crop_shape, tuple) and crop_shape[0]==crop_shape[1] and isinstance(crop_shape[0]), int) or crop_shape is None, f"'crop_shape': {crop_shape} should be either None or a tuple of int."
        assert isinstance(percentile, int) and 0<=percentile<=100, f"'percentile' should be a int between 0 and 100."
        assert isinstance(plot_boxes, bool), f"'plot':{plot_boxes} should be boolean. "
        assert isinstance(skip_cropped_objects, bool), f"'skip_cropped_objects' should be boolean."
        
        # attributes 
        self.W, self.H = image_shape
        self.root = root
        self.detect_dir = os.path.join(root, 'detection')
        self.cwd = '/Users/marco/yolov5_copy'
        self.save_folder = save_folder
        self.percentile = percentile
        self.plot = plot_boxes
        self.skip_cropped = skip_cropped_objects
        os.chdir(self.cwd)

        return


    def _get_last3_detect_dir(self) -> str:
        """ Get dir of the last detected images with YOLO (i.e. train, val test). """

        # get all dirs in /runs/detect:
        detect_dir = os.path.join(self.cwd, 'runs', 'detect')
        exps = [os.path.join(detect_dir, fold) for fold in os.listdir(detect_dir) if 'DS' not in fold]
        assert isinstance(exps, list) and len(exps) > 1, f"'exps' should be a list of length >= 3."

        # get 3 max nums in dirnames:
        nums = [fold.split('exp')[1] for fold in exps]
        nums = [int(fold) for fold in nums if len(fold)>0]
        nums = np.sort(np.array(nums))
        last3nums = nums[-3:].tolist()
        last3nums = [str(num) for num in last3nums]
        
        # get 3 respective dirs:
        last3folds = []
        for num in last3nums:
            found = [fold for fold in exps if num in os.path.split(fold)[1] ]
            if len(found) > 0:
                last3folds.extend(found)

        assert isinstance(last3folds, list) and len(last3folds) == 3,  f"'last3folders' should be a list of 3 dirs."

        print(f"Loading data from {self.detect_dir}: {[os.path.split(fold)[1] for fold in last3folds]}")

        return last3folds
    

    def _get_biggest_bbox(self, last3folders:list, ) -> int:
        """ Gets the biggest bounding boxes out of a folder. 
            last3folders:list = dirs of the last 3 detect folders where YOLO was applied. 
            img_shape:tuple = tuple of ints of the dims used by YOLO. """
        
        assert isinstance(last3folders, list) and len(last3folders) == 3,  f"'last3folders' should be a list of 3 dirs."
        assert os.path.isdir(last3folders[0]) and os.path.isdir(last3folders[0]) and os.path.isdir(last3folders[0]), f"'last3folders' should contain 3 valid pathdirs."

        # 1) getting all the labels:
        dirs = [os.path.join(dir, 'labels') for dir in last3folders]
        labels = []
        for dir in dirs:
            txt_files = glob(os.path.join(dir, '*.txt'))
            txt_files = [file for file in txt_files if 'DS' not in file]
            assert len(txt_files) > 2, f"Less than 2 labels found in {dir}"
            labels.extend(txt_files)

        # 2) get the max value out of all W, H:
        all_boxes = []
        for label in tqdm(labels, desc= f"Computing max bbox"): 
            w = self._get_info_from_label(label, 'w')
            h = self._get_info_from_label(label, 'h')
            all_boxes.extend(w)
            all_boxes.extend(h)
        all_boxes = np.array(all_boxes)

        perc90 = round(np.percentile(all_boxes, q = self.percentile), 2)

        # bbox dims distribution:
        if self.plot is True:
            plt.figure()
            sns.histplot(data = all_boxes)
            plt.xlabel("(norm) bbox dim")
            plt.title('Histogram bbox dim.')

        max_box = int(perc90 * self.W)
        print(f"max box: {(max_box, max_box)}")

        return max_box
    

    def _get_images(self) -> List[str]:
        """ Get all images for further cropping. """

        # collect images from train, val, test folders:
        dirnames = ['train', 'val', 'test']
        dirs = [os.path.join(self.detect_dir, fold, 'images') for fold in dirnames]
        all_images = []
        for dir in dirs:
            found_images = glob(os.path.join(dir, '*.png'))
            found_images = [image for image in found_images if 'DS' not in image]
            assert isinstance(found_images, list) and len(found_images) >= 2, f"Less than 2 images found in {dir}."
            all_images.extend(found_images)

        return all_images

    
    def _crop_images(self, images:list, max_box:int) -> None:
        """ Crops detected images with YOLO with the biggest bbox found. """

        assert isinstance(max_box, int) and 0<=max_box<max(self.H, self.W), f"Max bounding box = {max_box} (type={type(max_box)}) should be a int and should be less than the image size."

        for file in tqdm(images, desc = f"Cropping images"):

            # 0) check if already computed:
            if self._is_already_cropped(file):
                continue

            # 1) open:
            try:
                image = io.imread(file)
            except:
                print(f"Warning: couldn't open {file}. Skipped.")
                continue
            W, H, C = image.shape
            assert W == H == self.W == self.H, f"'image_shape' is set to {self.W, self.H}, but opened image has shape ({W, H}). "

            # 2) crop:
            label = self._img2lbl(file)
            if label is None: # i.e. no annotations
                continue
            all_xc, all_yc = self._get_info_from_label(label, 'xc'), self._get_info_from_label(label, 'yc')
            assert len(all_xc) == len(all_yc), f"'all_xc' (length {len(all_xc)}), but should have same length as all_yc (length {len(all_yc)}"


            for xc, yc in zip(all_xc, all_yc):
                
                xc, yc = int(self.W * xc), (self.H * yc)
                x0, x1, y0, y1 = int(xc - (max_box/2)), int(xc + (max_box/2)), int(yc - (max_box/2)), int(yc + (max_box/2))

                # 2.1) get crop x coords:
                if x0 > 0 and x1 < self.W:
                    crop_x0 = x0
                    crop_x1 = x1
                elif x0 >= 0 and x1 >= self.W: # if box would be out of the image
                    if self.skip_cropped is True:
                        continue
                    else:
                        crop_x0 = int(self.W - max_box)
                        crop_x1 = self.W
                elif x0 <= 0 and x1 <= self.W:
                    if self.skip_cropped is True:
                        continue
                    else:
                        crop_x0 = 0
                        crop_x1 = int(max_box)
                else:
                    raise Exception(f"x crop coords [{crop_x0, crop_x1}] exceed image x dim ({self.W}).")

                # 2.2) get crop y coords:
                if y0 >= 0 and y1 <= self.H:
                    crop_y0 = y0
                    crop_y1 = y1
                elif y0 >= 0 and y1 > self.H: # if box would be out of the image
                    if self.skip_cropped is True:
                        continue
                    else:
                        crop_y0 = int(self.H - max_box)
                        crop_y1 = self.H
                elif y0 < 0 and y1 <= self.H:
                    if self.skip_cropped is True:
                        continue
                    else:
                        crop_y0 = 0
                        crop_y1 = int(max_box)
                else:
                    raise Exception(f"y crop coords [{crop_y0, crop_y1}] exceed image y dim ({self.H}).")
                
                # 2.3) crop image: 
                crop_image = image[crop_y0:crop_y1, crop_x0:crop_x1]

                assert crop_image.shape[:2] == (max_box, max_box), f"{crop_image.shape}"

                self._save_crop(fp = file, crop = crop_image)

        return 


    def _save_crop(self, fp:str, crop:np.ndarray) -> None:
        """ Saves crops in the 'save_folder'. """

        assert os.path.isfile(fp), f"fp:{fp} is not a valid filepath."
        assert isinstance(crop, np.ndarray), f"'crop' is {type(crop)} but should be np.ndarray."

        fn = os.path.split(fp)[1]
        save_fp = os.path.join(self.save_folder, fn)
        crop = np.uint8(crop)
        io.imsave(fname = save_fp, arr = crop, check_contrast=False)

        return
    

    def _is_already_cropped(self, fp:str) -> bool:
        """ Checks whether there is already a cropped file with same name saved in save_folder. """

        assert os.path.isfile(fp), f"fp:{fp} is not a valid filepath."

        fn = os.path.split(fp)[1]
        save_fp = os.path.join(self.save_folder, fn)

        return os.path.isfile(save_fp)
    
    
    def __call__(self) -> None:
        """ Crops the inferred images with YOLO using the i_th percentile bbox defined in 'percentile'. """

        self.log.info("Cropping inferred images: ⏳")

        # 1) get last 3 infered folders (train, val, test):
        last3folders = self._get_last3_detect_dir()
        # 2) compute max (percentile) bbox:
        max_box = self._get_biggest_bbox(last3folders=last3folders)
        # 3) get all infered images:
        images = self._get_images()
        # 4) crop and save images:
        self._crop_images(images = images, max_box = max_box)

        self.log.info(f"Cropping inferred images done ✅. ")

        return


    ########################    HELPER FUNCTIONS    ########################

    def _img2lbl(self, file:str) ->str:
        """ Helper function to switch image filepath to its label filepath. """

        assert os.path.isfile(file) and '.png' in file, f"'file': {file} should be a valid filepath in .png format."
        file = file.replace('images', 'labels').replace('.png', '.txt')
        
        file = file if os.path.isfile(file) else None

        return file


    def _lbl2img(self, file:str) ->str:
        """ Helper function to switch image filepath to its label filepath. """

        assert os.path.isfile(file) and '.txt' in file, f"'file': {file} should be a valid filepath in .txt format."
        file = file.replace('labels', 'images').replace('.txt', '.png')
        assert os.path.isfile(file), f"'file': {file} is not a valid filepath."

        return file
    

    def _get_info_from_label(self, label:str, category:str) -> list:
        """ Reads a txt label and returns a dict like: {'class':[0,0,1,0...], 'xc':[], 'yc':[] , 'w':[] , 'h': []}"""
        
        assert os.path.isfile(label), ValueError(f"'label':{label} is not a valid filepath.")
        assert category in ['class', 'xc', 'yc', 'w', 'h'], ValueError(f"'category':{category} should be one of 'class', 'xc', 'yc', 'w', 'h'.")

        classes, xc, yc, w, h = [], [], [], [], []
        with open(label, 'r') as f:
            txt = f.readlines()
        for row in txt:
            words = row.split(' ')
            words = [word.replace('\n', '') for word in words]
            words = [word for word in words if word != '']
            if len(words)==0:
                continue
            words = [float(num) for num in words ]
            classes.append(words[0])
            xc.append(words[1])
            yc.append(words[2])
            w.append(words[3])
            h.append(words[4])

        df = {'class':classes, 'xc':xc, 'yc':yc, 'w':w, 'h':h}

        return df[category]
    
    
    
if __name__ == '__main__':
    root = '/Users/marco/datasets/muw_exps/'
    save_folder = '/Users/marco/butta'
    cropper = Cropper(root=root, 
                      save_folder=save_folder, 
                      image_shape=(4096,4096))
    # cropper._is_glom_whole(info = )
    cropper()


    # PROBLEMA: le crop fanno schifo perche' yolo detects gloms on patches -> tante volte 
    # c'e' solo un pezzo di glom che viene beccato da yolo -> crop e' su quel pezzetto
    # dovrei balzare tutti i casi in cui la crop e' sugli angoli e per quei casi non fare una crop
    # posso usare redundant tiling cosi dovrei avere sempre un'immagine in cui il glom e' intero?
