import os
from skimage import io, transform, color
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import utils_image



class GlomDataset(Dataset):
    """ Dataset for glomeruli images and masks. """


    def __init__(self, 
                img_dir,
                classes, 
                mapping,
                resize = False) -> None:
        
        mask_dir = img_dir.replace('images', 'masks')
        assert os.path.isdir(img_dir), f"'img_dir':{img_dir} is not a valid dirpath."
        assert isinstance(classes, int), f"'classes':{classes} should be an int."
        assert isinstance(resize, bool) or isinstance(resize, int), f"'resize':{resize} should be boolean or int."
        assert os.path.isdir(mask_dir), f"'mask_dir':{mask_dir} is not a valid dirpath."
        assert isinstance(mapping, dict), f"'mapping':{mapping} should be dict."

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.classes = classes
        self.resize = resize
        mask_dir = img_dir.replace('images', 'masks')
        self.imgs_fn = [file for file in os.listdir(img_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
        self.masks_fn = [file for file in os.listdir(mask_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
        self.imgs_fp = [os.path.join(img_dir, file) for file in self.imgs_fn]
        self.masks_fp = [os.path.join(mask_dir, file) for file in self.masks_fn]
        self.W, self.H, self.C = io.imread(self.imgs_fp[0]).shape
        self.mapping = mapping

        if len(self.imgs_fp) < len(self.masks_fp):
            print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional masks will be ignored.')
        elif len(self.imgs_fp) > len(self.masks_fp):
            print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional images will be ignored.')
        
        return 


    def __len__(self) -> int:
        return len(self.imgs_fp)


    def __getitem__(self, idx) -> dict:

        assert os.path.isfile(self.imgs_fp[idx]), f"idx={idx} -> image:{self.imgs_fp[idx]} is not a valid filepath."
        assert os.path.isfile(self.masks_fp[idx]), f"idx={idx} -> mask:{self.masks_fp[idx]} is not a valid filepath."

        img_fp = self.imgs_fp[idx]
        mask_fp = self.masks_fp[idx]

        img = io.imread(img_fp, as_gray=False)
        mask = io.imread(mask_fp, as_gray=False)

        mask = np.expand_dims(mask, axis = 2) if mask.ndim <=2 else mask
        img = color.rgba2rgb(img)*255 if img.shape[2] == 4 else img
        mask = color.rgba2rgb(mask)*255 if mask.shape[2] == 4 else mask

        if self.resize is not False:
            img = transform.resize(img, (self.resize, self.resize))
            mask = transform.resize(mask, (self.resize, self.resize))

        if self.classes <= 1:
            mask = T.Grayscale()(mask)
            mask = torch.Tensor(mask).int()
            mask = mask / 255
            mask = torch.where(mask == 0., 0., 1.)

        elif self.classes >= 2: # map colors from RGB to classes
            mask = utils_image.map_RGB2coloridx(mask = mask, mapping = self.mapping, classes = self.classes)


        assert isinstance(mask, torch.Tensor), f"mask is type {type(mask)}, but should be type torch.Tensor"
        assert mask.shape == torch.Size([self.classes, self.W, self.H]), f"Mask shape is {mask.shape}, but should be {torch.Size([self.classes, self.W, self.H])}"
        assert mask.max() <= 1.0 and mask.min() >= 0, f"Mask range is in ({mask.min()}, {mask.max()}), but should be (0,1)"  

        data = {'image': img, 'mask': mask, 'fname': os.path.split(mask_fp)[1] }
        

        return data
    
