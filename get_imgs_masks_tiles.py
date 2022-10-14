from openslide import OpenSlide
import geojson
from requests import patch
from skimage import io, measure, draw
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from patchify import patchify
from PIL import Image
import os
from warnings import warn



def get_tiles(slide_folder: str, out_folder: str, tile_shape: int = 2048, tile_step: int = None) -> None:
    ''' Generates image and mask tiles from slides/annotations.
        NB: requires slides files and their annotations to be in the same folder. '''


    if tile_step is None:
        tile_step = tile_shape

    slides_fp = [os.path.join(slide_folder, slide) for slide in os.listdir(slide_folder) if 'tiff' in slide]

    out_folder_masks = os.path.join(out_folder, 'masks')
    out_folder_imgs = os.path.join(out_folder, 'images')
    if not os.path.isdir(out_folder_masks):
        os.makedirs(out_folder_masks)
    if not os.path.isdir(out_folder_imgs):
        os.makedirs(out_folder_imgs)

    for fp in slides_fp:

        # check if slide already computed:
        fn = os.path.split(fp)[1].replace('.tiff', '')
        computed = [file for file in os.listdir(out_folder_imgs) if fn in file ]
        if len(computed) > 0:
            print(f'Slide: {fn}.tiff already computed, skipping. ')
            continue

        # read slide
        try:
            slide = OpenSlide(fp)
        except:
            print(f" WARNING: Couldn't open slide: {fn}.tiff, skipping." )
            continue
        H, W = slide.dimensions
        print(f'Reading slide: {os.path.split(fp)[1]}')
        slide = slide.read_region(location = (0,0), level = 0, size = slide.dimensions)
        slide = np.array(slide)


        # read annotations
        fp = fp.replace('tiff', 'json')
        with open(fp, 'r') as f:
            data = geojson.load(f)
            f.close()


        # generating masks:
        slide_mask = np.zeros((W, H, 1))
        for i, glom in enumerate(tqdm(data, desc = f'Generating slide mask')): 
            vertices = glom['geometry']['coordinates']
            vertices = vertices.pop()
            vertices = [ (y, x) for (x,y) in vertices]
            mask = draw.polygon2mask((W, H), vertices) # for each glom coords draw polygon in mask
            slide_mask[mask == 1] = 255


        # visually test mask: 
        plt.figure()
        plt.imshow(slide)
        plt.show()
        plt.figure()
        plt.imshow(slide_mask)
        plt.show()


        # patchifying masks:
        patches_masks= patchify(slide_mask, (tile_shape, tile_shape, 1), step = tile_step )
        patches_masks = patches_masks.squeeze()
        patches_imgs = patchify(slide, (tile_shape, tile_shape, 3), step = tile_step)
        patches_imgs = patches_imgs.squeeze()
        print('Patches for images and masks generated.')


        # saving masks and images
        fn = os.path.split(fp)[1]
        fn = fn.replace('.json', '')
        print(f'Saving mask tiles in {out_folder_masks}:')
        for i in tqdm(range(patches_masks.shape[0]), leave = False):
            for j in tqdm(range(patches_masks.shape[1])):
                fn_mask = fn + f'_{i}_{j}.png'
                fn_mask = os.path.join(out_folder_masks, fn_mask)
                pil_img = Image.fromarray(patches_masks[i, j])
                pil_img = pil_img.convert('RGB')
                pil_img.save(fn_mask)
        print(f'Saving image tiles in {out_folder_imgs}:')
        for i in tqdm(range(patches_imgs.shape[0]), leave = False):
            for j in tqdm(range(patches_imgs.shape[1])):
                fn_img = fn + f'_{i}_{j}.png'
                fn_img = os.path.join(out_folder_imgs, fn_img)
                pil_img = Image.fromarray(patches_imgs[i, j])
                pil_img = pil_img.convert('RGB')
                pil_img.save(fn_img)
        
        #TODO: speed up by putting everything in same loop



if __name__ == '__main__':
    slide_folder = '/Users/marco/hubmap/yolo_data/wsis/train/'
    out_folder = '/Users/marco/hubmap/tiles'

    get_tiles(slide_folder, out_folder)


