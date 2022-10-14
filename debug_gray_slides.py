
from openslide import deepzoom, open_slide
import os 
import matplotlib.pyplot as plt
import numpy as np


def debug(root: str):

    fns =  [file for file in os.listdir(root) if '.svs' in file and '1e2425f28' in file ]
    fps =  [os.path.join(root, file) for file in fns]
    for fn, fp in zip(fns, fps):
        print(f'Opening {fn}')
        slide = open_slide(fp)
        slide_vendor = slide.detect_format(fp)
        dims = slide.dimensions
        # region = slide.read_region(location = (0,0), level = 0, size = slide.dimensions)
        rgba_image = slide.get_thumbnail(size = slide.dimensions)
        print(slide_vendor)
        print(dims)
        # print(region.size)
        print(rgba_image.mode)
        print(rgba_image.size)
        rgb_image = np.array(rgba_image.convert('RGB'))


        plt.figure()
        plt.imshow(rgba_image)
        plt.show()

        plt.figure()
        plt.imshow(rgb_image)
        plt.show()

    return


if __name__ == '__main__':
    debug(root = '/Users/marco/Downloads/debug')
