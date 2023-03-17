import os
from skimage import io, color, feature
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as ttf
import torch
from torchvision.io import read_image
import pytorch_lightning as plh
import matplotlib.pyplot as plt 
import numpy as np
from typing import List
from glob import glob
import cv2


def mask_to_polygon(mask: np.array, report: bool = False) -> List[int]:
    """ Converts mask to polygon """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for object in contours:
        coords = []

        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        polygons.append(coords)
    
    if report:
        print(f"Number of points = {len(polygons[0])}")
    
    return np.array(polygons).ravel().tolist()
    


def get_missing_converted(wsi_folder:str):
    """ Given a root folder with a list of wsi files, it checks that all slides 
        have a label, a json ROI file and a txt file. """
    
    wsis = glob(os.path.join(wsi_folder, '*.tif'))
    missing_txt = []
    missing_geojgson = [] 
    for fp in wsis: 
        label1 = fp.replace('.tif', '.gson')
        label2 = fp.replace('.tif', '.mrxs.gson')
        if os.path.isfile(label1):
            label = os.path.basename(label1)
        elif os.path.isfile(label2):
            label = os.path.basename(label2)
        else:
            print(f"❌ fp:{os.path.basename(fp)} has no label.")
        # txt label:
        txt_label = fp.replace('.tif', '.txt')
        if not os.path.isfile(txt_label):
            print(f"❌ wsi:{os.path.basename(fp)} doesn't have a txt label. Convert using Converter(convert_from='gson_wsi_mask').")
            missing_txt.append(fp)
        # geojson roi:
        geojson_label = fp.replace('.tif', '.geojson')
        if not os.path.isfile(geojson_label):
            print(f"❌ wsi: {os.path.basename(fp)} doesn't have a geojson label.")
            missing_geojgson.append(fp)            
        # txt sample label:
        txt_samples = glob(os.path.join(wsi_folder, f"{os.path.basename(fp.split('.')[0])}*sample*.txt"))
        # print(os.path.join(wsi_folder, f"{os.path.basename(fp)}*sample*.txt"))
        if len(txt_samples)<1:
            print(f"❌ wsi: {os.path.basename(fp)} doesn't have a txt sample label. \n     Convert ROI annotations using Converter(convert_from='gson_wsi_mask').")
            missing_geojgson.append(fp)   
    
    



    return

def test_get_missing_converted(): 
    wsi_folder = '/Users/marco/Downloads/muw_slides'
    get_missing_converted(wsi_folder=wsi_folder)
    return


if __name__ == '__main__':
    test_get_missing_converted()


def map_RGB2coloridx(mask: np.ndarray, mapping: dict, classes: int, one_hot:bool = True) -> torch.Tensor:
    """ Maps RGB colors to indexes for further one-hot encoding. """

    assert isinstance(mask, np.ndarray), f"mask type is {type(mask)}, but should be np.ndarray"
    assert isinstance(mapping, dict), f"'mapping' is type {type(mapping)} but should be dict."
    assert isinstance(classes, int), f"'classes' should be type int but is type {classes}."
    assert all(value in [0,255] for value in np.unique(mask)), f"Unique values in 'mask' are {np.unique(mask)}, but should be in [0,255] "
    assert mask.shape[2]==3 or mask.shape[2]==1, f"Channel is {mask.shape[2]}, but should be RGB --> should be = 3."
    assert isinstance(one_hot, bool), f"'one_hot' should be boolean."

    if mask.shape[2] == 3:
        target = mask
        H, W, C = target.shape
        target = torch.from_numpy(target)
        target = target.permute(2, 0, 1).contiguous()
        mask = torch.empty(H, W, dtype=torch.long)
        for k in mapping:
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    else:
        mask = torch.tensor(mask/255, dtype = torch.long).squeeze()

    assert mask.ndim == 2, f"'mask.ndim' == {mask.ndim}, but should be = 2."
    assert mask.max() < classes, f"Max value of the index tensor (output of RGB mapping) should be smaller than number of classes."

    # to torch tensor C,W,H:
    if one_hot is True:
        mask = F.one_hot(mask, num_classes = classes) if one_hot else mask
        mask = mask.permute(2, 0, 1)

    return mask


def get_corners(image: np.ndarray) -> np.ndarray:

    coords = feature.corner_peaks(feature.corner_harris(image), min_distance=2, threshold_rel=0.)
    plt.figure()
    plt.imshow(image)
    plt.plot(coords[:, 1], coords[:, 0], 'ob', markersize=4)
    plt.show()

    return


def mask_to_YOLOsegmentation(file:str) -> None:
    """ Converts and saves a mask image to a txt file segmentation label suitable for YOLO segmentation. """

    # 1) open txt file: 
    with open(file, "r") as f:
        text = f.read(file)
    
    # 2) 

    return





# if __name__ == '__main__':

#     test_get_corners()