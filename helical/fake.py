import os, shutil, sys 
from glob import glob 
import random


def make_fake_dataset(dataset, map_classes:dict):  # map_classes = desired classes
    """ Given a path to a dataset like cnn_dataset -> train, val, test -> false_positives, item , 
        makes a fake dataset like cnn_dataset -> wsi, tiles -> train, val, test -> Glo-Healthy, Glo-unhealthy -> images,labels """
        
    # for image in old_images:
    old_images = glob(os.path.join(dataset, '*', '*', '*.jpg'))
    old_images = [img for img in old_images if os.path.split(os.path.dirname(img))[1] != 'false_positives']

    # create new folds for new dataset
    new_dataset_root = os.path.join(os.path.dirname(dataset), 'fake_dataset')
    img_formats = ['wsi', 'tiles']
    datasets = ['train', 'val', 'test']
    classes = list(map_classes.keys())
    for fmt in img_formats:
        for _set in datasets:
            for clss in classes: 
                os.makedirs(os.path.join(new_dataset_root, fmt, _set, clss), exist_ok=True)
    
    # copy images to fake dataset
    for image in old_images: 

        # pick random class
        new_class = random.choice(list(map_classes.keys()))
        src=image
        dst = os.path.join(new_dataset_root, 'tiles', random.choice(datasets) , 
                           new_class, os.path.basename(image).split('_crop')[0] + ".png") # os.path.split(os.path.dirname(os.path.dirname(image)))[1]
        if not os.path.isfile(dst):
            shutil.copy(src=src, dst=dst)

    return




if __name__ == '__main__': 
    dataset = '/Users/marco/helical_tests/test_cnn_trainer/cnn_dataset'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1}
    make_fake_dataset(dataset=dataset, map_classes=map_classes)