from patchify import patchify
import openslide
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from tqdm import tqdm 
import warnings

def slide2patches(fp):
    ''' Reads a slide, patchify it. '''
    
    try:
        slide = openslide.OpenSlide(fp)
    except:
        warnings.warn(f'Couldn t open file: {fp}. Skipping. ')
        return

    dims = slide.dimensions
    print(f'Slide shape: {dims}')
    slide = slide.read_region(location = (0,0), level = 0, size= dims).convert("RGB")
    slide = np.array(slide)
    patches = patchify(slide, (2048, 2048, 3), step = 2048 )
    print(f'Patches shape: {patches.shape}')
    

    patches = patches[:, :, 0, ...]
    print(patches.shape)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            fname = fp.replace('.tiff',f'_{i}_{j}.png')
            pil_img = Image.fromarray(patches[i, j])
            pil_img.save(fname)

    return

def slides2patches(folder):
    ''' Takes a folder and patchify all slides in it. '''

    all_files = os.listdir(folder)
    all_files = [os.path.join(folder, item) for item in all_files if '.tiff' in item]

    for file in all_files:
        slide2patches(file)


    return

def del_empty_imgs(folder):
    '''Takes a folder and deletes empty tiles (black or white). '''

    
    all_files = os.listdir(folder)
    all_files = [os.path.join(folder, item) for item in all_files if 'png' in item]
    n_files = len(all_files)

    # del all emtpy and truncated files:
    n_empty_files = 0
    n_couldntopen_files = 0
    for file in all_files:
        try:
            tile = io.imread(file)
            if (tile.mean() <= 0.1 or tile.mean() >= 248) and tile.var() <= 30:
                n_empty_files += 1
                os.remove(file)
        except:
            n_couldntopen_files += 1
            os.remove(file)
            warnings.warn(f'Couldnt open file {file}. Deleted it.')

    
    print(f'Total files: {n_files}. Deleted: {n_empty_files + n_couldntopen_files} (empty: {n_empty_files}), truncated files: {n_couldntopen_files}. Remaining files: {n_files - n_couldntopen_files - n_empty_files}.')


    return  


if __name__ == '__main__':
    folder = '/Users/marco/Downloads/train/'
    print(folder)
    slides2patches(folder)
    # del_empty_imgs(folder) 