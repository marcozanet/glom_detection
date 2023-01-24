
import os
vipsbin = r'c:\vips-dev-8.13\bin'
add_dll_dir = getattr(os, 'add_dll_directory', None)
if callable(add_dll_dir):
    add_dll_dir(vipsbin)
else:
    os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))

import pyvips
from pyvips import Image

input = Image.new_from_file('/Users/marco/Downloads/new_source/aaa6a05cc.tiff')
print('yay')

print(type(input))


# a = input.tiffsave(filename ='/Users/marco/converted/201216192_09_SFOG/201216192_09_SFOG_Wholeslide_Default_Extended_pyramidal.tiff', 
#                compression = 'jpeg',
#                 Q = 90,
#                 tile = True, 
#                 tile_width = 256, 
#                 tile_height = 256, 
#                 pyramid = True)

