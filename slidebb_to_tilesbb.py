import json
from PIL import Image
import numpy as np
import geojson
import os
import openslide
from tqdm import tqdm 
import warnings
from skimage import io, draw
import matplotlib.pyplot as plt


def get_bb_from_txt(fp, shape_patch):
    ''' Open txt and get bb numbers. '''

    # open slide and get original dims
    slide_fp = fp.replace('_boxes.txt', '.tiff')
    slide = openslide.OpenSlide(slide_fp)
    W, H = slide.dimensions
    print(f'Slide dims: {(W,H)}')

    # read the slide annotation and write annotations for each patch
    with open(fp, 'r') as f:
        text = f.readlines()
        f.close()
    
    assign_dict = {}

    for r, row in enumerate(text):
        items = row.split(sep = ',')
        xc, yc, box_w, box_h = [float(num) for num in items[1:]]
        w, h = shape_patch, shape_patch
        i = int(xc // w) 
        j = int(yc // h) 
        # print(f'i: {i}, j: {j}')
        img_fp = fp.replace('_boxes.txt', f'_{j}_{i}.png') # img that contains the center of that glom
        xc = xc % w  
        yc = yc % h
        print(f'Glom should be at location: {xc, yc}')
        
        txt_fp = img_fp.replace('.png', '.txt')
        text = f'0, {xc}, {yc}, {box_w}, {box_h}\n'

        # append if file exists, otherwise write:
        if os.path.exists(txt_fp):
            mode = 'a' 
        else:
            mode = 'w' 

        # save txt file:
        with open(txt_fp, mode) as f:
            f.write(text)

            f.close()

        # visual test case
            # img = io.imread(img_fp)
            # print('ok')
            # start= ((xc % 2048) - box_h/2, (yc % 2048) - box_w/2)
            # extent= (box_h, box_w)
            # row, col = draw.rectangle(start= start, extent= extent)
            # plt.figure()    
            # plt.imshow(img)
            # # plt.plot(row, col, '-y')
            # plt.show()
            # raise Exception()

    return


''' 1 - from wsi json segmentation annootations to wsi bounding boxes annotations txt format. 
    2 - from wsi bb txt to tiles bb txt. '''



if __name__ == '__main__':
    fp = '/Users/marco/Downloads/train/1e2425f28_boxes.txt'
    get_bb_from_txt(fp, shape_patch = 2048)
    



