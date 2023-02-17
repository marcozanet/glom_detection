import os 
import glob 
from typing import Literal
import shutil
from tqdm import tqdm


def pair_up(root:str, 
            save_folder:str,
            slide_format:Literal['.tif_pyr.tif'], 
            label_format:Literal['.mrxs.gson'], 
            stain = "SFOG", 
            copy:bool = True,
            clear_others:bool = False, 
            verbose:bool=False): 
    """ Pairs up files from MUW -> selects what files have a correspondent annotation file, 
        with option of deleting the other ones. """

    ALLOWED_SLIDE_FORMATS =  '.tif_pyr.tif'
    ALLOWED_LABEL_FORMATS =  '.mrxs.gson'

    assert os.path.isdir(root), f"'root' is not a valid dirpath."
    assert os.path.isdir(save_folder), f"'save_folder' is not a valid dirpath."
    assert slide_format in ALLOWED_SLIDE_FORMATS, f"'slide_format' should be one of {ALLOWED_SLIDE_FORMATS}. "
    assert label_format in ALLOWED_LABEL_FORMATS, f"'label_format' should be one of {ALLOWED_LABEL_FORMATS}. "
    assert isinstance(copy, bool), f"'safe_copy':{copy}."
    assert isinstance(verbose, bool), f"'verbose':{verbose}."

    # get all labels:
    labels = glob.glob(os.path.join(root, f'*{label_format}'))
    labels = [label for label in labels if os.stat(label).st_size > 1000 and stain in label ]
    useless_labels = [file for file in glob.glob(os.path.join(root, f'*{label_format}')) if file not in labels ]
    labels = list(set(labels))

    # get potential slide matches for existing labels:
    pot_pairs = [(label.replace('.mrxs.gson','_Wholeslide_Default_Extended.tif_pyr.tif'), label) for label in labels]
    pot_pairs = [(os.path.join(root, os.path.basename(label.split('.')[0]), os.path.basename(slide)), label) for slide, label in pot_pairs]
    matched = []
    missing_slides = []
    for slide, label in pot_pairs:
        if os.path.isfile(slide) and os.path.isfile(label):
            matched.append((slide, label))
        elif (not os.path.isfile(slide)) and os.path.isfile(label):
            missing_slides.append(slide)

    # get all slides:
    slides = []
    for file in glob.glob(os.path.join(root, f'**/*{slide_format}'), recursive=True):
        if stain in file:
            slides.append(file)
    # get potential (labels) matches for existing slides
    pot_labels_fn = [os.path.split(os.path.dirname(slide))[1] + label_format for slide in slides]
    pot_labels_fp = [os.path.join(root, label) for label in pot_labels_fn]
    missing_labels = [label for label in pot_labels_fp if label not in labels ]

    print(f"Matched pairs: {len(matched)}")
    print(f"Found annotations, but not pyramidal slides (maybe only not converted slide exists?) for {len(missing_slides)} pairs.")
    print(f"Found {len(missing_labels)} slides without annotations.") # either missing or empty file

    # get slides not pyramidal:
    not_pyramidal = []
    for file in glob.glob(os.path.join(root, f'**/*tif'), recursive=True):
        if stain in file:
            not_pyramidal.append(file)
    not_pyramidal = [os.path.pardir(slide) for slide in not_pyramidal if slide not in slides]


    # copy/move matched slides,labels to save_folder:
    if clear_others is True:
        for file in not_pyramidal:
            os.remove(path=file)
            print(f"Deleting not pyramidal slide: {file}")
        for file in missing_labels:
            if os.path.isfile(file):
                print(f"Deleting missing label {file}")
                os.remove(path=file)
        for file in useless_labels:
            if os.path.isfile(file):
                print(f"Deleting useless label {file}")
                os.remove(path=file)
            

    
    if copy is True:
        for slide, label in tqdm(matched):
            shutil.copy(src=label, dst=os.path.join(save_folder, os.path.basename(label)).replace('.mrxs', ''))
            shutil.copy(src=slide, dst=os.path.join(save_folder, os.path.basename(slide)).replace('_Wholeslide_Default_Extended', '').replace('.tif_pyr', '') )
    else:
        for slide, label in matched:
            shutil.move(src=label, dst=os.path.join(save_folder, os.path.basename(label)).replace('.mrxs', ''))
            shutil.move(src=slide, dst=os.path.join(save_folder, os.path.basename(slide)).replace('_Wholeslide_Default_Extended', '').replace('.tif_pyr', '') )


    return


def test_pair_up():
    root = '/Users/marco/converted_30_01_23'
    slide_format = '.tif_pyr.tif'
    label_format = '.mrxs.gson'
    save_folder = '/Users/marco/Downloads/muw'

    pair_up(root=root, 
            save_folder=save_folder,
            slide_format=slide_format, 
            label_format=label_format,
            copy=True,
            clear_others=True,
            verbose=True)



    return

if __name__ == '__main__':
    test_pair_up()