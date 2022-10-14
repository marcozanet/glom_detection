# import json
from PIL import Image
import numpy as np
import geojson
import os
# The path can also be read from a config file, etc.
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


def read_slide(fp):
    ''' Reads the slide and returns shape. '''

    print('SLIDE OPEN')
    wsi = openslide.OpenSlide(fp)
    W, H = wsi.dimensions
    print(f'dimensions: {W, H}')

    return W, H


def get_bounding_boxes(W, H, fp, returned = 'yolo'):
    ''' Iterates through gloms and gets the bounding box from segmentation annotations. '''
    
    # read file
    with open(fp, 'r') as f:
        data = geojson.load(f)
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
            #print(f'polygon vertices: {vertices}')
            
            # saving outer coords (bounding boxes) for each glom
            x_min = 10000000000
            y_min = 10000000000
            x_max = 0
            y_max = 0
            for i, xy in enumerate(vertices[0]):
                # print(xy)
                x = xy[0]
                y = xy[1]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max 
                y_min = y if y < y_min else y_min

            # print(f'x_min: {x_min}, x_max: {x_max}')
            # print(f'y_min: {y_min}, y_max: {y_max}')

            if x_max > W:
                raise Exception()
            if y_max > H:
                raise Exception()

            new_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
            x_c =  (x_max + x_min) / 2 
            y_c = (y_max + y_min) / 2  
            # print(f'xc: {x_c} = x_max: {x_max} + x_min:{x_min} / 2 = {(x_max + x_min / 2)} -> / W: {W} = {(x_max + x_min / 2) / W} ')
            box_w, box_y = (x_max - x_min) , (y_max - y_min) 
            boxes.append([0, x_c, y_c, box_w, box_y])
            # print(boxes)

        return_obj = boxes if returned == 'yolo' else new_coords
        
        # Closing file
        f.close()

    # print('----------------------------')
    # print(f'bounding boxes: {new_coords}')
    # print('----------------------------')

    return return_obj

 
def copy_data(new_coords, fp):
    ''' Copies the same annotations with bounding boxes instead of segmentation vertices.'''

    # copy old data with the replaced boxes instead of polygon
    with open(fp, 'r') as f:
        data = geojson.load(f)
    
        # Iterating through gloms
        gloms = 0
        for i, glom in enumerate(data):
            gloms += 1
            glom['geometry']['coordinates'] = [new_coords[i]]
        
        # Closing file
        f.close()

    print(f'Num gloms: {gloms}')

    return data


def write_geojson(data, fp):
    """ Dumps the copied data into a json file. """

    with open(fp.replace('.json', '_boxes.geojson').replace('hub_bb', 'bb'), 'w') as f:
        data = geojson.dump(data, f)
        f.close()
    print('Saved.')


def write_txt(data, fp):
    """ Dumps the copied data into a txt file. """

    with open(fp.replace('.json', '_boxes.txt').replace('hub_bb', 'bb'), 'w') as f:
        text = ''
        for values in data:
            text += str(values) + '\n'
        text = text.replace('[', '').replace(']', '')
        f.write(text)

        f.close()

    # print(text)
    print('Saved.')


def convert_segm2bb(source_folder):
    """ Reads slide, gets bounding boxes and saves in a new file. """

    files = [os.path.join(source_folder, file) for file in os.listdir(source_folder) if '.json' in file ]
    wsis = [file.replace('json', 'tiff') for file in files]

    # print(f'files: {files}')
    skipped = 0
    for fp, wsi in tqdm(zip(files, wsis)):
        try:
            W, H = read_slide(wsi)
            returned_obj = get_bounding_boxes(W, H, fp)
            # data = copy_data(new_coords=returned_obj, fp = fp)
            # write_geojson(data, fp)
            write_txt(returned_obj, fp)
        except:
            skipped += 1
            print(f'Cannot open, skipping slide. Total skipped: {skipped}')
            continue




if __name__ == '__main__':
    source_folder = '/Users/marco/downloads/train' 
    convert_segm2bb(source_folder)
    # read_slide(fp  = '/Users/marco/Downloads/train/cb2d976f4.tiff')


""" L'errore e' dovuto al for, se runni il singolo file non esce nessun errore. """