import time 
import torch
import os, shutil, sys
from glob import glob
from tqdm import tqdm
from cnn_feat_extract_loaders import CNNDataLoaders
from cnn_assign_class_crop import CropLabeller
from cnn_splitter import CNNDataSplitter
from typing import List
import numpy as np

def prepare_data(cnn_root_fold:str, map_classes:dict, batch:int, num_workers:int, 
                 yolo_root:str, exp_folds:List[str], resize_crops:bool): 
    """ Prepares data for feature extraction: puts all images in the same folder 
        (regardless of trainset, valset, testset) and gets the dataloader to be used by the model."""
    
    assert 'false_positives' in map_classes.keys(), f"'false_positives' missing in 'map_classes'. "
    cnn_root = os.path.join(cnn_root_fold, 'cnn_dataset')

    # Assign true classes back to crops out of yolo:
    print("Labelling crops from YOLO:")
    print("-"*10)
    for exp_fold in exp_folds:
        labeller = CropLabeller(root_data=yolo_root, exp_data=exp_fold, map_classes=map_classes, resize = False)
        labeller()

    # Creating Dataset and splitting into train, val, test:
    print("Creating Dataset and splitting into train, val, test:")
    print("-"*10)
    cnn_processor = CNNDataSplitter(src_folds=exp_folds, map_classes=map_classes, yolo_root=yolo_root, 
                                    dst_root=cnn_root, resize=resize_crops)
    cnn_processor()


    # Putting images all in same fold and extracting features:
    print("Create Dataset for Feature Extraction:")
    print("-"*10)
    feat_extract_fold = os.path.join(cnn_root_fold, 'feat_extract')
    os.makedirs(feat_extract_fold, exist_ok=True)
    # get all images: 
    images = glob(os.path.join(cnn_root, '*', '*', '*.jpg')) # get all images 
    class_fold_names = glob(os.path.join(cnn_root, '*', '*/'))
    class_fold_names = set([os.path.split(os.path.dirname(fp))[1] for fp in class_fold_names])
    # print(class_fold_names)

    assert len(images)>0, f"No images like {os.path.join(cnn_root, '*', '*', '*.jpg')}"
    # create folds:
    for fold in class_fold_names: 
        os.makedirs(os.path.join(feat_extract_fold, fold), exist_ok=True)
    # fill fold:
    for img in tqdm(images, desc="Filling 'feature_extract'"): 
        clss_fold_name = os.path.split(os.path.dirname(img))[1]
        dst = os.path.join(feat_extract_fold, clss_fold_name, os.path.basename(img))
        if not os.path.isfile(dst):
            shutil.copy(src=img, dst=dst)
    assert len(os.listdir(feat_extract_fold))>0, f"No images found in extract fold: {feat_extract_fold}"
    dataloader_cls = CNNDataLoaders(root_dir=feat_extract_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
    dataloader = dataloader_cls()

    return  dataloader


def feature_extraction(model, dataloader, cnn_root_fold) -> None:
    """ Extracts features and saves them in cnn_root_fold. """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Evaluating model")
    print('-' * 10)
    save_feats = lambda save_fp, feats: np.save(save_fp, feats.numpy())

    # make folds for classes: 
    class_fold_names = glob(os.path.join(cnn_root_fold, 'cnn_dataset', '*', '*/'))
    class_fold_names = set([os.path.split(os.path.dirname(fp))[1] for fp in class_fold_names])
    for clss in class_fold_names: 
        os.makedirs(os.path.join(cnn_root_fold, 'feat_extract',  clss, 'feats'), exist_ok=True)

    # Feature extraction:
    for i, data in enumerate(tqdm(dataloader, "Feat extraction")):
        model.train(False)
        model.eval()
        inputs, labels, fp = data
        inputs.to(device), labels.to(device) # to gpu
        outputs = model(inputs)
        fp=fp[0] # because it's batched otherwise
        save_fp = os.path.join(os.path.dirname(fp), 'feats', os.path.basename(fp).replace('.jpg', '.npy'))
        save_feats(save_fp=save_fp, feats=outputs.data )
        del inputs, labels, outputs 
        torch.cuda.empty_cache()


    return



if __name__ == "__main__": 

    cnn_root_fold = '/Users/marco/helical_tests/test_cnn_processor/test_crossvalidation'
    # map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
    # batch = 1
    # num_workers = 0
    # dataloader = prepare_data(cnn_root_fold=cnn_root_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
    # features = feature_extraction(model, dataloader)
