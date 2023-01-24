import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from skimage.feature import blob_doh, blob_dog, blob_log
from skimage.color import rgb2gray
from PIL import ImageOps
import matplotlib.pyplot as plt
from math import sqrt


# questo tissue detector potrebbe essere usato in due fasi: 
# 1) per creare ROI (rectangle) on multiple samples 
# 2) per vedere dato un tile se c'e' dentro del tessuto e in caso
#   contrario elimino tanti tile bianchi, altrimenti sono troppi 


def detect_tissue(fp:str):
    slide = openslide.OpenSlide(fp)
    print(slide.level_dimensions)
    W, H = slide.dimensions
    image = slide.read_region(location = (0,0), level = 2, size = (W,H )).convert('RGB').convert()
    print(image)
    image_gray = rgb2gray(image=image)
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)


    # show 
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
            'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()

    return


def test_detect_tissue():

    fp = '/Users/marco/Downloads/muw_slides/201420222_09_SFOG.tif'
    detect_tissue(fp)


    return


if __name__ == '__main__':
    test_detect_tissue()