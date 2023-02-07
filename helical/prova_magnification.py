import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from glob import glob
from typing import List
import warnings
from patchify import patchify
import numpy as np 
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io, draw
import cv2
import geojson
from typing import Literal
import random


fp = '/Users/marco/Downloads/try_train/detection/wsi/train/images/200813457_A_09_SFOG.tif'

slide = openslide.OpenSlide(fp)


for level in range(3):
    region = slide.read_region(location = (5000,5000) , level = level, size= (4096, 4096)).convert("RGB")

    region.show()

