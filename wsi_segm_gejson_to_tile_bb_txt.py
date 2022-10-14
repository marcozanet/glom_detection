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


def read_slide(fp):
    ''' Reads slide and returns dims. '''
    try:
        wsi = openslide.OpenSlide(fp)
        W, H = wsi.dimensions
    except:
        print(f"Couldn't open {fp}")

    return W, H


def get_bounding_boxes(W, H, fp, returned = 'yolo'):
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


def convert_segm2boundbox(source_folder, convert_to = 'yolo'):
    """ Reads slide, gets bounding boxes and saves in a new file. """

    files = [os.path.join(source_folder, file) for file in os.listdir(source_folder) if '.json' in file ]
    wsis = [file.replace('json', 'tiff') for file in files]

    # print(f'files: {files}')
    skipped = 0
    for fp, wsi in tqdm(zip(files, wsis)):
        try:
            W, H = read_slide(wsi)
        except:
            skipped += 1
            # print(f'Cannot open, skipping slide. Total skipped: {skipped}')
            continue

        returned_obj = get_bounding_boxes(W, H, fp)
        if convert_to == 'qupath': # i.e. geojson vertices
            data = copy_data(new_coords=returned_obj, fp = fp)
            write_geojson(data, fp)
        elif convert_to == 'yolo': # i.e. txt bounding boxes
            write_txt(returned_obj, fp)

    
    print(f'{len(files) - skipped} text WSI annotations created. Skipped {skipped} files that did not open.')

    return


def patchify_slide_annotations(folder, shape_patch = 2048):
    ''' Patchifies all slides annotations in the folder in txt format to patch annotations in txt format. '''

    txt_files = [os.path.join(folder, file) for file in os.listdir(folder) if '_boxes.txt' in file]
    for file in tqdm(txt_files):
        patchify_slide_annotation(fp = file, shape_patch = shape_patch)

    print(f'Patch annotations created for {len(txt_files)} WSI annotations. ')
    return

def patchify_slide_annotation(fp, shape_patch):
    ''' Patchifies one slide annotation in txt format to patch annotations in txt format. '''
    try:
        # Read WSI:
        slide_fp = fp.replace('_boxes.txt', '.tiff')
        slide = OpenSlide(slide_fp)
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
    ''' Converts segmentation WSI annotations to patch annotations in txt format (suitable for YOLO). '''

    # 1 - convert segmentation WSI annotations in geojson format to WSI bounding boxes segmentations in txt format (YOLO format)
    convert_segm2boundbox(source_folder=folder, convert_to= 'yolo')
    # 2 - turn WSI bounding box annotations in txt format to patch bounding box annotations in txt format
    patchify_slide_annotations(folder = folder, shape_patch= shape_patch)

    return

if __name__ == '__main__':
    convert(folder = '/Users/marco/hubmap/yolo_data/wsi/test', shape_patch= 2048)