from openslide import OpenSlide
import os

folder = '/Users/marco/downloads/train'

slides = [os.path.join(folder, file) for file in os.listdir(folder) if 'tiff' in file]
print(slides)
for i, file in enumerate(slides):
    try:
        slide = OpenSlide(file)
    except:
        print(f"Could't open file {os.path.split(file)[1]} ")
print('Done')
 