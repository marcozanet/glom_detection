import os 

import geojson
from typing import Literal



def split_multisample_annotation(txt_file:str, multiple_loc_file:str) -> None:
    """ Given a WSI txt (not normalised) annotation, it splits the annotation file 
        into one file for each sample within the slide."""

    
    assert os.path.isfile(txt_file), f"'label_file':{txt_file} is not a valid filepath."
    assert os.path.isfile(multiple_loc_file), f"'label_file':{multiple_loc_file} is not a valid filepath."
    assert txt_file.split(".")[-1] == 'txt', f"'txt_file':{txt_file} should have '.txt' format. "


    with open(txt_file, 'r') as f:
        rows = f.readlines()
    
    for row in rows:
        class_label, xc, yc, w, h = row.split(' ')
        # print(f"class_label, xc, yc, w, h: {class_label, xc, yc, w, h}")


    return

    

def test_split_multisample_annotation():
    
    txt_file = '/Users/marco/Downloads/another/8242609fa.txt'
    multiple_loc_file = '/Users/marco/Downloads/converted_test/200415540_09_SFOG_msample_image.geojson'
    split_multisample_annotation(txt_file=txt_file, multiple_loc_file=multiple_loc_file)
    return

if __name__ == '__main__':
    
    test_split_multisample_annotation()