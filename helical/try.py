import json
import geojson
import openslide
from glob import glob 
import os 
import random
from tqdm import tqdm

def remove_empty_images(empty_perc: float = 0.1, min_mem_size:int = 100_000):

    dst_dir = '/Users/marco/Downloads/try_train'
    task ='detection'
    # print(os.path.join(dst_dir, task, 'tiles', '*', 'labels', '*.txt' ))
    # 1) get all image tiles from the train, val, test
    train_images = glob(os.path.join(dst_dir, task, 'tiles', 'train', 'images', '*.png' ))
    assert all(['wsi' not in file for file in train_images]), f"All selected files should be tile images, not wsi images."
    assert all(['images' in file for file in train_images]), f"All selected files should be tile images, not labels."
    assert len(train_images)> 0, f"No image found."
    # 2) get all label tiles from train, val, test
    train_labels = glob(os.path.join(dst_dir, task, 'tiles', 'train', 'labels', '*.txt' ))
    train_labels = [label for label in train_labels if 'test' not in train_labels]
    assert all(['wsi' not in file for file in train_labels]), f"All selected files should be tile labels, not wsi labels."
    assert all(['labels' in file for file in train_labels]), f"All selected files should be tile labels, not images."
    assert len(train_labels)> 0, f"No label found."

    # 3) collect images without a label (i.e. empty)
    train_empty = [image for image in train_images if image.replace('images', 'labels').replace('png', 'txt') not in train_labels]
    assert all([not os.path.isfile(image.replace('images', 'labels').replace('png', 'txt')) for image in train_empty])

    # 4) from this files keep an empty_perc and delete the other ones:
    train_full = [image for image in train_images if image.replace('images', 'labels').replace('png', 'txt') in train_labels]
    # train_images, trin_labels = [image for image in images if 'val' in image], [label for label in labels if 'val' in labels]
    
    # check if empty percantage already <= empty_perc:
    if (len(train_empty)/len(train_images)) <= empty_perc:
        print(f"Empty perc of images: {round(len(train_empty)/len(train_images), 2)} already <= {empty_perc}. ")
        return

    # delete random empty images:
    # print(f"train images: {len(train_images)}")
    # print(f'train_full: {len(train_full)}')
    # print(f"train empty: {len(train_empty)}")
    n_train_wished =  int(len(train_full) * (1+ empty_perc))
    n_del_train =  len(train_images) - n_train_wished 
    train_del_imgs = random.sample(train_empty, n_del_train )
    assert all([not os.path.isfile(image.replace('images', 'labels').replace('png', 'txt')) for image in train_del_imgs])
    for file in tqdm(train_del_imgs):
        os.remove(file)
    final_train_images = glob(os.path.join(dst_dir, task, 'tiles', 'train', 'images', '*.png' ))
    print(f"Removed {n_del_train} empty images. Train images: {len(train_images)} -> {len(final_train_images)} .")
    assert len(final_train_images) == n_train_wished, f"Wished: {n_train_wished}, obtained: {len(final_train_images)}"


    return


def print_properties(file):

    slide = openslide.OpenSlide(file)
    print(slide.dimensions)
    slide_props = slide.properties
    print(slide_props)
    vendor = slide_props.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, 0)
    print(vendor)


    return

if __name__ == '__main__':
    # file = '/Users/marco/converted_2/199608490_09_SFOG/199608490_09_SFOG_Wholeslide_Default_Extended.tif'
    # print_properties(file)
    remove_empty_images()



