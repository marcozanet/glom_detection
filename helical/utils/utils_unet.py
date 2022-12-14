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
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# class GlomDataset(Dataset):

#     def __init__(self, img_dir, classes = 1, resize = False, transform=None, target_transform=None):

#         mask_dir = img_dir.replace('images', 'masks')
#         self.imgs_fn = [file for file in os.listdir(img_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
#         self.masks_fn = [file for file in os.listdir(mask_dir) if '.png' in file and 'DS' not in file and 'preds' not in file]
#         self.imgs_fp = [os.path.join(img_dir, file) for file in self.imgs_fn]
#         self.masks_fp = [os.path.join(mask_dir, file) for file in self.masks_fn]
#         self.resize = resize
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.classes = classes

#         if len(self.imgs_fp) < len(self.masks_fp):
#             print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional masks will be ignored.')
#         elif len(self.imgs_fp) > len(self.masks_fp):
#             print(f'Warning: len images: {len(self.imgs_fp)}, len masks: {len(self.masks_fp)}. Additional images will be ignored.')
#         # self.transform = transform
#         # self.target_transform = target_transform

#     def __len__(self):
#         return len(self.imgs_fp)

#     def __getitem__(self, idx):
#         img_fp = self.imgs_fp[idx]
#         mask_fp = self.masks_fp[idx]
#         try:
#             img = read_image(img_fp)
#         except:
#             raise TypeError(f'{img_fp}')
#         try:
#             mask = read_image(mask_fp)
#         except:
#             raise TypeError(f'{mask_fp}')

#         if self.resize is not False:
#             img = ttf.resize(img, (self.resize, self.resize))
#             mask = ttf.resize(mask, (self.resize, self.resize))

#         if self.classes == 0 or self.classes == 1:
#             mask = T.Grayscale()(mask)
#             mask = torch.Tensor(mask).int()
#             mask = mask / 255
#             # print(mask.unique())
        
#         elif self.classes == 2:
        
#             target = mask
#             target = target.permute(1, 2, 0).numpy()
#             # plt.imshow(target) # Each class in 10 rows
#             H, W, C = target.shape
#             # print(target.shape)
#             # Create mapping
#             # Get color codes for dataset (maybe you would have to use more than a single
#             # image, if it doesn't contain all classes)
#             target = torch.from_numpy(target)
#             # print(colors)
#             target = target.permute(2, 0, 1).contiguous()
#             # print(target.shape)
#             mapping = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}
#             # mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
#             # print(f"mapping:{mapping}")

#             mask = torch.empty(H, W, dtype=torch.long)
#             for k in mapping:
#                 # Get all indices for current class
#                 idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
#                 validx = (idx.sum(0) == 3)  # Check that all channels match
#                 mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
                
#             # mask = F.one_hot(mask, num_classes = 3)
#             # mask = mask.permute(2, 0, 1)
#             mask = mask.unsqueeze(0)
#         fname = os.path.split(mask_fp)[1]
        
#         if self.classes == 1:
#             mask = torch.where(mask == 0., 0., 1.)
#         assert type(mask) == torch.Tensor, f"mask is type {type(mask)}, but should be type torch.Tensor"
#         assert mask.shape == torch.Size([self.classes, 512, 512]), f"Mask shape is {mask.shape}, but should be {torch.Size([self.classes, 512, 512])}"
#         assert mask.max() <= 1.0 and mask.min() >= 0, f"Mask range is in ({mask.min()}, {mask.max()}), but should be (0,1)"  
#         if self.classes == 1:
#             uniques = mask.unique().tolist()
            
#             assert all(elem in [0., 1.] for elem in uniques), f"out_classes = {self.classes}, but mask unique values are: {uniques}"

#         # print(f"MASK SHAPE IN DATASET WITHOUT ONE HOT: {mask.shape}")
#         # if self.transform:
#         #     image = self.transform(image)
#         # if self.target_transform:
#         #     label = self.target_transform(label)
#         data = {'image': img, 'mask': mask, 'fname': fname }
        
#         return data

# def split_sets(img_dir: str, split_ratio = [0.7, 0.15, 0.15]):
#     ''' From YOLO cropped images and masks, splits them into subdirs for train, val, test. '''


#     # get images and masks
#     mask_dir = img_dir.replace('images', 'masks')
#     imgs_fn = [file for file in os.listdir(img_dir) if 'png' or 'jpg' in file]
#     masks_fn = [file for file in os.listdir(mask_dir) if 'png' or 'jpg' in file]
#     if len(imgs_fn) > len(masks_fn):
#         print(f'Warning: found {len(imgs_fn)} images and {len(masks_fn)} masks. Ignoring {len(imgs_fn) - len(masks_fn)} images. ')
#         imgs_fn = [file for file in imgs_fn if file in masks_fn]
#     elif len(masks_fn) > len(imgs_fn):
#         print(f'Warning: found {len(imgs_fn)} images and {len(masks_fn)} masks. Ignoring {len(masks_fn) - len(imgs_fn)} masks. ')
#         masks_fn = [file for file in masks_fn if file in imgs_fn]

#     print(f'For check: {len(imgs_fn)} images and {len(masks_fn)} masks. ')
#     imgs_fp = [os.path.join(img_dir, file) for file in imgs_fn]
#     masks_fp = [os.path.join(mask_dir, file) for file in masks_fn]
#     # only images with corresponding masks are to be included:


#     train_idx, test_idx = int(split_ratio[0] * len(imgs_fp)), int((split_ratio[0] + split_ratio[1]) * len(imgs_fp))
#     train_imgs, val_imgs, test_imgs = imgs_fp[:train_idx], imgs_fp[train_idx:test_idx], imgs_fp[test_idx:]
#     train_masks, val_masks, test_masks = masks_fp[:train_idx], masks_fp[train_idx:test_idx], masks_fp[test_idx:]
#     subsets = [train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks]


#     # create dirs:
#     root = os.path.split(img_dir)[0]
#     train_dir_imgs, val_dir_imgs, test_dir_imgs = os.path.join(root, 'train', 'images'), os.path.join(root, 'val', 'images'), os.path.join(root, 'test', 'images')
#     train_dir_masks, val_dir_masks, test_dir_masks = os.path.join(root, 'train', 'masks'), os.path.join(root, 'val', 'masks'), os.path.join(root, 'test', 'masks')
#     subdirs = [train_dir_imgs, val_dir_imgs, test_dir_imgs, train_dir_masks, val_dir_masks, test_dir_masks]
#     sets = dict(zip(subdirs, subsets))
#     for subdir in subdirs:
#         if not os.path.isdir(subdir):
#             os.makedirs(subdir)


#     # filling imgs and masks into subdirs:
#     for subdir, subset in sets.items():
#         print(subset)
#         for file in subset:
#             src = file
#             print(src)
#             dst = os.path.join( subdir , os.path.split(file)[1])
#             print(f'src: {file}')
#             print(f'dst: {dst}')
#             os.rename(src, dst)


#     return


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


def get_last_model(path_to_exps: str):
    """ Returns path to the last trained model. """

    files = os.listdir(path_to_exps)
    nums = [file.split('_')[1] for file in files if 'version' in file]
    if len(nums) == 0:
        raise Exception(f"No 'version' folder found in the experiment folder '{path_to_exps}'. " )
    nums = np.array([int(file) for file in nums if len(file) > 0])
    last = str(nums.max())
    last = [file for file in files if last in file][0]
    version_path = os.path.join(path_to_exps, last)
    try:
        hparams_file = [os.path.join(version_path, file) for file in os.listdir(version_path) if 'hparams' in file][0]
    except:
        raise Exception(f"No hparams.yaml found in {path_to_exps}")
    last = os.path.join(version_path, 'checkpoints')
    last = [file for file in os.listdir(last) if 'ckpt' in file][0]
    last = os.path.join(version_path, 'checkpoints', last)
    
    return last, hparams_file


def get_testloader(img_dir: str, classes = 3, batch_size = 4, num_workers = 0):
    testset = GlomDataset(img_dir=img_dir, classes = classes) # TODO TODO TODO FAR SI CHE SIA UN PARAMETRO 
    print(f"Test size: {len(testset)} images.")
    # n_cpu = os.cpu_count()
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_dataloader


def write_hparams_yaml(hparams_file: str, hparams: dict):

    hparams = json.dumps(hparams)
    if not os.path.isfile(hparams_file):
        raise ValueError(f"'{hparams_file}' is not a valid yaml file path.")

    with open(hparams_file, 'w') as f:
        f.write(hparams)
        f.close()

    return


def pred_mask2colors(pred_mask: torch.Tensor, n_colors = int):
    """ Converts predicted mask to colors. """

    if n_colors == 3:
        color_map = {0: 0, 1: 127, 2: 255}
    elif n_colors == 2:
        color_map = {0:0, 1:255}
    else:
        raise ValueError(f"n_colors = {n_colors}, but admitted n_colors are 2 or 3.")

    for key, value in color_map.items():
        pred_mask[pred_mask == key] = value 



    return pred_mask



def test_glomdataset():

    dataset = GlomDataset(img_dir = '/Users/marco/hubmap/training/train/model_train/unet/images',
                          classes = 1, 
                          resize = False, 
                          transform= None)
    data = dataset[3]

    return


def test_write_yaml():


    hparams = {'arch' : 'unet',
        'encoder_name': 'resnet34', 
        'encoder_weights': 'imagenet', 
        'in_channels' : 3,
        'out_classes': 3,
        'activation' : None}
    
    system = 'mac'
    if system == 'windows':
        test_folder = r'D:\marco\zaneta-tiles-pos0_02\test'
        path_to_exps = r'C:\marco\biopsies\zaneta\lightning_logs'
    elif system == 'mac':
        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs'
        test_folder = '/Users/marco/zaneta-tiles-pos0_02/test'

    last, hparams_file = get_last_model(path_to_exps= path_to_exps)
    write_hparams_yaml(hparams_file= hparams_file, hparams = hparams)

    return


def save_masks(batch: dict, pr_masks: torch.Tensor, classes: int, show = False):

    # get last preds_n folder
    # save_par_fold = os.path.join(os.path.split(os.getcwd())[0], 'unet_tests')
    # os.makedirs(save_par_fold, exist_ok=True)
    # folds = np.array([int(fold.split('_')[1]) for fold in os.listdir(save_par_fold) if os.path.isdir(os.path.join(save_par_fold, fold))])
    # # print(f'FOLDSSSSSSSSS: {folds}')
    # num = 0 if len(folds) == 0 else folds.max() + 1
    save_folder = os.path.join(os.path.split(os.getcwd())[0], 'unet_tests', f'preds')
    os.makedirs(save_folder, exist_ok=True)


    # save preds:
    for i, (image, gt_mask, pr_mask, fname) in enumerate(zip(batch["image"], batch["mask"], pr_masks, batch["fname"])):
        
        image, gt_mask, pr_mask = image.cpu(), gt_mask.cpu(), pr_mask.cpu()
        # print(f"Image: {fname}")

        # print(f"GT uniques: {gt_mask.unique()}, pred uniques: {pr_mask.unique()} ")
        if classes > 1:
            gt_mask = pred_mask2colors(pred_mask = gt_mask, n_colors= classes)
            pr_mask = pred_mask2colors(pred_mask= pr_mask, n_colors = classes)
        else:
            pr_mask = np.array(pr_mask.squeeze()) * 255
            gt_mask = np.array(gt_mask.squeeze()) * 255
            pr_mask = np.uint8(pr_mask)
            gt_mask = np.uint8(gt_mask)

        os.makedirs(os.path.join(save_folder, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'plots'), exist_ok=True)

        io.imsave(fname = os.path.join(save_folder, 'masks', fname ), arr = pr_mask, check_contrast=False)
        
        # print(np.unique(pr_mask))
        # print(np.unique(gt_mask))

        # plot:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask) # just squeeze classes dim, because we have only one class
        title = f"Prediction_{i}"
        plt.title(title)
        plt.axis("off")

        if show is True:
            plt.show()

        # if i % save_plot_every == 0:
        plot_fname = f'plot_{fname}'
        fig.savefig(os.path.join(save_folder, 'plots', plot_fname ))
        plt.close()

if __name__ == '__main__':
    test_write_yaml()

