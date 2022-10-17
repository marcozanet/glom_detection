import os
from skimage import io 
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as ttf
import torch
from torchvision.io import read_image
import pytorch_lightning as pl
import numpy as np


class GlomDataset(Dataset):

    def __init__(self, img_dir, classes = 1, resize = False, transform=None, target_transform=None):

        mask_dir = img_dir.replace('images', 'masks')
        self.imgs_fn = [file for file in os.listdir(img_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
        self.masks_fn = [file for file in os.listdir(mask_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
        self.imgs_fp = [os.path.join(img_dir, file) for file in self.imgs_fn]
        self.masks_fp = [os.path.join(mask_dir, file) for file in self.masks_fn]
        self.resize = resize
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.classes = classes

        if len(self.imgs_fp) < len(self.masks_fp):
            print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional masks will be ignored.')
        elif len(self.imgs_fp) > len(self.masks_fp):
            print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional images will be ignored.')
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs_fp)

    def __getitem__(self, idx):
        img_fp = self.imgs_fp[idx]
        mask_fp = self.masks_fp[idx]
        try:
            img = read_image(img_fp)
        except:
            raise TypeError(f'{img_fp}')
        try:
            mask = read_image(mask_fp)
        except:
            raise TypeError(f'{mask_fp}')

        if self.resize is not False:
            img = ttf.resize(img, [3, self.resize, self.resize])
            mask = ttf.resize(mask, [3, self.resize, self.resize])

        if self.classes == 0 or self.classes == 1 or self.classes == 2:
            mask = T.Grayscale()(mask)

        mask = mask /255
        # img = img /255
        # mask = mask /255
        # print(img.shape)
        # print(mask.shape)
        fname = os.path.split(mask_fp)[1]

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        data = {'image': img, 'mask': mask, 'fname': fname }
        
        return data

def split_sets(img_dir: str, split_ratio = [0.7, 0.15, 0.15]):
    ''' From YOLO cropped images and masks, splits them into subdirs for train, val, test. '''


    # get images and masks
    mask_dir = img_dir.replace('images', 'masks')
    imgs_fn = [file for file in os.listdir(img_dir) if 'png' or 'jpg' in file]
    masks_fn = [file for file in os.listdir(mask_dir) if 'png' or 'jpg' in file]
    if len(imgs_fn) > len(masks_fn):
        print(f'Warning: found {len(imgs_fn)} images and {len(masks_fn)} masks. Ignoring {len(imgs_fn) - len(masks_fn)} images. ')
        imgs_fn = [file for file in imgs_fn if file in masks_fn]
    elif len(masks_fn) > len(imgs_fn):
        print(f'Warning: found {len(imgs_fn)} images and {len(masks_fn)} masks. Ignoring {len(masks_fn) - len(imgs_fn)} masks. ')
        masks_fn = [file for file in masks_fn if file in imgs_fn]

    print(f'For check: {len(imgs_fn)} images and {len(masks_fn)} masks. ')
    imgs_fp = [os.path.join(img_dir, file) for file in imgs_fn]
    masks_fp = [os.path.join(mask_dir, file) for file in masks_fn]
    # only images with corresponding masks are to be included:


    train_idx, test_idx = int(split_ratio[0] * len(imgs_fp)), int((split_ratio[0] + split_ratio[1]) * len(imgs_fp))
    train_imgs, val_imgs, test_imgs = imgs_fp[:train_idx], imgs_fp[train_idx:test_idx], imgs_fp[test_idx:]
    train_masks, val_masks, test_masks = masks_fp[:train_idx], masks_fp[train_idx:test_idx], masks_fp[test_idx:]
    subsets = [train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks]


    # create dirs:
    root = os.path.split(img_dir)[0]
    train_dir_imgs, val_dir_imgs, test_dir_imgs = os.path.join(root, 'train', 'images'), os.path.join(root, 'val', 'images'), os.path.join(root, 'test', 'images')
    train_dir_masks, val_dir_masks, test_dir_masks = os.path.join(root, 'train', 'masks'), os.path.join(root, 'val', 'masks'), os.path.join(root, 'test', 'masks')
    subdirs = [train_dir_imgs, val_dir_imgs, test_dir_imgs, train_dir_masks, val_dir_masks, test_dir_masks]
    sets = dict(zip(subdirs, subsets))
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            os.makedirs(subdir)


    # filling imgs and masks into subdirs:
    for subdir, subset in sets.items():
        print(subset)
        for file in subset:
            src = file
            print(src)
            dst = os.path.join( subdir , os.path.split(file)[1])
            print(f'src: {file}')
            print(f'dst: {dst}')
            os.rename(src, dst)


    return


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        print(f'inputs: {inputs.shape}, targets: {targets.shape}')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def get_last_model(path_to_exps = '/Users/marco/yolo/lightning_logs'):
    """ Returns path to the last trained model. """

    files = os.listdir(path_to_exps)
    nums = [file.split('_')[1] for file in files if 'version' in file]
    nums = np.array([int(file) for file in nums if len(file) > 0])
    last = str(nums.max())
    last = [file for file in files if last in file][0]
    version_path = os.path.join(path_to_exps, last)
    hparams_file = [os.path.join(version_path, file) for file in os.listdir(version_path) if 'hparams' in file][0]
    last = os.path.join(version_path, 'checkpoints')
    last = [file for file in os.listdir(last) if 'ckpt' in file][0]
    last = os.path.join(version_path, 'checkpoints', last)

    return last, hparams_file

if __name__ == '__main__':
    # GlomDataset(img_dir = '/Users/marco/hubmap/unet_data/train/images')
    # save_imgs_folder = '/Users/marco/hubmap/unet_data/images'
    # split_sets(img_dir = save_imgs_folder
    get_last_model()
