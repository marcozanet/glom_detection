"""  Prepare data to be trained using the YOLO model: splitting of the images into train, val, test folders. """
# TODO: add also other data preparation 

from typing import Any
from PIL import Image
import numpy as np
import geojson
import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from tqdm import tqdm 
import warnings
from tqdm import tqdm
from patchify import patchify


###################################################################
############         FOLDER PATHS PREPARATION           ###########
###################################################################

def split_wsi_yolo(data_folder: str, new_root: str, ratio: float = [0.7, 0.15, 0.15]):
    """ Given the download folder with WSIs and annotations in JSON format, moves them 
        into a new folder and splits them in train, val, test. """
    skip_check = False


    # TODO aggiungi il caso in cui siano aggiunte NUOVE slide. 


    # create dirs
    wsis_folder_train = os.path.join(new_root, 'wsis', 'train')
    wsis_folder_val = os.path.join(new_root, 'wsis', 'val')
    wsis_folder_test = os.path.join(new_root, 'wsis', 'test')
    tiles_folder_train = os.path.join(new_root, 'tiles', 'train')
    tiles_folder_val = os.path.join(new_root, 'tiles', 'val')
    tiles_folder_test = os.path.join(new_root, 'tiles', 'test')
    if not os.path.isdir(tiles_folder_train):
        os.makedirs(tiles_folder_train)
    if not os.path.isdir(tiles_folder_val):
        os.makedirs(tiles_folder_val)
    if not os.path.isdir(tiles_folder_test):
        os.makedirs(tiles_folder_test)
    if not os.path.isdir(tiles_folder_train):
        os.makedirs(wsis_folder_train)
        skip_check = True
    if not os.path.isdir(wsis_folder_val):
        os.makedirs(wsis_folder_val)
        skip_check = True
    if not os.path.isdir(wsis_folder_test):
        os.makedirs(wsis_folder_test)
        skip_check = True

    # check that wsis aren't already splitted:




    # get slides fps:
    wsi_fns = [file for file in os.listdir(data_folder) if '.tiff' in file and 'DS' not in file]
    wsi_fps = [os.path.join(root, file) for file in wsi_fns]
    print(f"Slides found: {wsi_fps}. ")

    # split the WSI names between train, val, test
    n_slides = len(wsi_fns)
    train_idx, val_idx = int(ratio[0] * n_slides), max(int((ratio[0] + ratio[1]) * n_slides), 1)
    train_wsis, val_wsis, test_wsis = wsi_fns[:train_idx], wsi_fns[train_idx:val_idx], wsi_fns[val_idx:]
    train_masks = [file.replace('tiff', 'json') for file in train_wsis]
    val_masks = [file.replace('tiff', 'json') for file in val_wsis]
    test_masks = [file.replace('tiff', 'json') for file in test_wsis]
    slides = [train_wsis, val_wsis, test_wsis]
    dirs = [wsis_folder_train, wsis_folder_val, wsis_folder_test]
    masks = [train_masks, val_masks, test_masks]


    # move slides into new folders:
    for slides, dir in zip(slides, dirs):
        for slide in slides:
            src = os.path.join(data_folder, slide)
            dst = os.path.join(dir, slide)
            os.rename(src = src, dst = dst)
    for masks, dir in zip(masks, dirs):
        for mask in masks:
            src = os.path.join(data_folder, mask)
            dst = os.path.join(dir, mask)
            os.rename(src = src, dst = dst)

    return wsis_folder_train, wsis_folder_val, wsis_folder_test
 

###################################################################
############     ANNOTATION TXT AND BOXES CREATION     ############
###################################################################

def read_slide(fp):
    ''' Reads slide and returns dims. '''
    try:
        wsi = openslide.OpenSlide(fp)
        W, H = wsi.dimensions
    except:
        print(f"Couldn't open {fp}")

    return W, H


def get_bb(W, H, fp, returned = 'yolo'):
    ''' Iterates through gloms and gets the bounding box from segmentation geojson annotations. '''
    
    # read file
    with open(fp, 'r') as f:
        data = geojson.load(f)
        f.close()
        
    gloms = 0
    new_coords = []
    boxes = []

    # saving outer coords (bounding boxes) for each glom
    x_min = 10000000000
    y_min = 10000000000
    x_max = 0
    y_max = 0

    # access polygon vertices of each glom
    for glom in data:
        gloms += 1
        vertices = glom['geometry']['coordinates']
        
        # saving outer coords (bounding boxes) for each glom
        x_min = 10000000000
        y_min = 10000000000
        x_max = 0
        y_max = 0
        for i, xy in enumerate(vertices[0]):
            x = xy[0]
            y = xy[1]
            x_max = x if x > x_max else x_max
            x_min = x if x < x_min else x_min
            y_max = y if y > y_max else y_max 
            y_min = y if y < y_min else y_min

        if x_max > W:
            raise Exception()
        if y_max > H:
            raise Exception()

        x_c =  (x_max + x_min) / 2 
        y_c = (y_max + y_min) / 2  
        box_w, box_y = (x_max - x_min) , (y_max - y_min)
        new_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]) 
        boxes.append([0, x_c, y_c, box_w, box_y])

    return_obj = boxes if returned == 'yolo' else new_coords

    return return_obj

 
def copy_data(new_coords, fp):
    ''' Copies the same annotations and replaces bounding boxes to segmentation vertices.'''

    # copy old data with the replaced boxes instead of polygon
    with open(fp, 'r') as f:
        data = geojson.load(f)
    
        # Iterating through gloms
        gloms = 0
        for i, glom in enumerate(data):
            gloms += 1
            glom['geometry']['coordinates'] = [new_coords[i]]
        f.close()

    return data


def write_geojson(data, fp):
    """ Dumps the copied data into a json file. """

    with open(fp.replace('.json', '_boxes.geojson').replace('hub_bb', 'bb'), 'w') as f:
        data = geojson.dump(data, f)
        f.close()

    return


def write_txt(data, fp):
    """ Dumps the copied data into a txt file. """

    with open(fp.replace('.json', '_boxes.txt').replace('hub_bb', 'bb'), 'w') as f:
        text = ''
        for values in data:
            text += str(values) + '\n'
        text = text.replace('[', '').replace(']', '')
        f.write(text)
        f.close()

    return


def get_wsi_bb(source_folder,  convert_to = 'yolo'):
    """ Reads slide, gets bounding boxes and saves in a new file. """

    files = [os.path.join(source_folder, file) for file in os.listdir(source_folder) if '.json' in file ]
    wsis = [file.replace('json', 'tiff') for file in files]

    # print(f'files: {files}')
    skipped = 0
    for fp, wsi in zip(files, wsis):
        try:
            W, H = read_slide(wsi)
        except:
            skipped += 1
            # print(f'Cannot open, skipping slide. Total skipped: {skipped}')
            continue

        returned_obj = get_bb(W, H, fp)
        if convert_to == 'qupath': # i.e. geojson vertices
            data = copy_data(new_coords=returned_obj, fp = fp)
            write_geojson(data, os.path.join(fp))
        elif convert_to == 'yolo': # i.e. txt bounding boxes
            write_txt(returned_obj, os.path.join(fp))

    print(f'{len(files) - skipped} text WSI annotations created. Skipped {skipped} files that did not open.')

    return


def get_tiles_bb(folder, shape_patch = 2048):
    ''' Patchifies all slides annotations in the folder in txt format to patch annotations in txt format. '''

    txt_files = [os.path.join(folder, file) for file in os.listdir(folder) if '_boxes.txt' in file]
    for file in tqdm(txt_files):
        get_one_tiles_bb(fp = file, shape_patch = shape_patch)

    print(f'Patch annotations created for {len(txt_files)} WSI annotations. ')

    return

def get_one_tiles_bb(fp, shape_patch):
    ''' Patchifies one slide annotation in txt format to patch annotations in txt format. '''
    try:
        # Read WSI:
        slide_fp = fp.replace('_boxes.txt', '.tiff')
        slide = openslide.OpenSlide(slide_fp)
    except:
        print('patchify skipped')
        return

    # Get BB from txt file:
    with open(fp, 'r') as f:
        text = f.readlines()
        f.close()
    
    # for each glom, find corresponding patch:
    for row in text:
        items = row.split(sep = ',')
        xc, yc, box_w, box_h = [float(num) for num in items[1:]]
        w, h = shape_patch, shape_patch
        i = int(xc // w) 
        j = int(yc // h) 
        img_fp = fp.replace('_boxes.txt', f'_{j}_{i}.png') # img that contains the center of that glom

        # convert WSI coords to patch coords:
        xc = xc % w  
        yc = yc % h
        #print(f'Glom patch coords: {xc, yc}')
        
        # write patch annotation txt file:
        txt_fp = img_fp.replace('.png', '.txt')
        text = f'0 {xc/2048} {yc/2048} {box_w/2048} {box_h/2048}\n'

        if os.path.exists(txt_fp):
            mode = 'a' #append if file exists
        else:
            mode = 'w' # write new if does not

        # save txt file:
        with open(txt_fp, mode) as f:
            f.write(text)

            f.close()

    return


def convert(folder, shape_patch = 2048):
    ''' Converts WSI segmentation annotations to patch annotations in txt format (suitable for YOLO). '''

    # 1 - convert segmentation WSI annotations in geojson format to WSI bounding boxes in txt format (YOLO format)
    get_wsi_bb(source_folder=folder, convert_to= 'yolo')
    # 2 - turn WSI bounding box annotations in txt format to patch bounding box annotations in txt format
    get_tiles_bb(folder = folder, shape_patch= shape_patch)

    return


def move_tiles(src_folder: str, mode: str, dst_folder = False ):
    """ Takes tiles annotations and images created and moves them into their train, val, test folders. """
    

    if mode == 'train':
        
        files = [file for file in os.listdir(src_folder) if 'txt' in file and 'boxes' not in file and 'tiff' not in file and 'json' not in file]
        for file in files:
            src = os.path.join(src_folder, file)
            dst = os.path.join(src_folder, 'tiles', 'bb', file)
            os.rename(src, dst)

    elif mode == 'test':

        if dst_folder is False:
            raise Exception(f"In testing mode 'dst_folder' is to be specified. ")

        slides = [file.split('.')[0] for file in os.listdir(src_folder) if '.tiff' in file]
        for slide in slides:
            tiles_bb = [file for file in os.listdir(src_folder) if '.txt' in file and 'boxes' not in file and slide in file]
            for file in tiles_bb:
                src = os.path.join(src_folder, file)
                dst = os.path.join(dst_folder, 'predictions', slide, 'tiles', 'bb', file)
                os.rename(src, dst)
            

    return

###################################################################
############     PATCHIFICATION FROM SLIDES TO TILES   ############
###################################################################

def get_one_tiles(fp):
    ''' Reads a slide, patchify it. '''
    
    try:
        slide = openslide.OpenSlide(fp)
    except:
        warnings.warn(f'Couldn t open file: {fp}. Skipping. ')
        return

    dims = slide.dimensions
    slide = slide.read_region(location = (0,0), level = 0, size= dims).convert("RGB")
    slide = np.array(slide)
    patches = patchify(slide, (2048, 2048, 3), step = 2048 )
    

    patches = patches[:, :, 0, ...]
    for i in tqdm(range(patches.shape[0])):
        for j in range(patches.shape[1]):
            fname = fp.replace('.tiff',f'_{i}_{j}.png')
            pil_img = Image.fromarray(patches[i, j])
            pil_img.save(fname)

    return

def get_tiles(folder):
    ''' Takes a folder and patchify all slides in it. '''

    all_files = os.listdir(folder)
    all_files = [os.path.join(folder, item) for item in all_files if '.tiff' in item]
    print(f"Patchifying {len(all_files)} slides. ")
    for file in all_files:
        get_one_tiles(file)

    return




if __name__ == "__main__":
    root = '/Users/marco/hubmap'
    images_folder = '/Users/marco/hubmap/unet_data/images'
    # split_data_yolo(root, ratio = [0.7, 0.15, 0.15], images_folder= images_folder)

    # split_wsi_yolo(data_folder= '/Users/marco/Downloads/train',
    #                ratio = [0.7, 0.15, 0.15], 
    #                new_root='/Users/marco/hubmap/yolo_data')
    
    convert(folder = '/Users/marco/hubmap/yolo_data/wsi/test', shape_patch= 2048)