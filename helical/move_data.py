
import os 
import shutil


def move_slides_for_tiling(wsi_folder:str, slide_format:str):
    """ Temporarly moves slides for tiling in labels folder. """
    assert os.path.isdir(wsi_folder), f"'wsi_folder':{wsi_folder} is not a valid dirpath."
    
    subdirs = ['train', 'val', 'test']
    folders = [os.path.join(wsi_folder, subdir, 'images') for subdir in subdirs]

    for slides_folder in folders:
        assert os.path.isdir(slides_folder), f"'slides_folder':{slides_folder} is not a valid dirpath."

        # slides_fp = [os.path.join(slides_folder, file) for file in slides_folder]
        trainvaltest_dir = os.path.dirname(slides_folder)
        # print(wsi_dir)
        for slide_fn in [file for file in os.listdir(slides_folder) if slide_format in file]: 

            src = os.path.join(trainvaltest_dir, 'images', slide_fn)
            dst = os.path.join(trainvaltest_dir, 'labels', slide_fn)

            assert os.path.isfile(src), f"'src':{src} is not a valid filepath."
            dst_dir = os.path.join(trainvaltest_dir, 'labels')
            assert os.path.isdir(dst_dir), f"dst dir: {dst_dir} doesn't exist."

            shutil.move(src = src, dst = dst)

    return

def move_slides_back_from_tiling(wsi_folder:str, slide_format:str):
    """ Moves slides from label folder back to images folder after tiling . """
    assert os.path.isdir(wsi_folder), f"'wsi_folder':{wsi_folder} is not a valid dirpath."
    
    subdirs = ['train', 'val', 'test']
    folders = [os.path.join(wsi_folder, subdir, 'labels') for subdir in subdirs]

    for slides_folder in folders:
        assert os.path.isdir(slides_folder), f"'slides_folder':{slides_folder} is not a valid dirpath."

        # slides_fp = [os.path.join(slides_folder, file) for file in slides_folder]
        trainvaltest_dir = os.path.dirname(slides_folder)
        # print(wsi_dir)
        for slide_fn in [file for file in os.listdir(slides_folder) if slide_format in file]: 

            src = os.path.join(trainvaltest_dir, 'labels', slide_fn)
            dst = os.path.join(trainvaltest_dir, 'images', slide_fn)

            assert os.path.isfile(src), f"'src':{src} is not a valid filepath."
            dst_dir = os.path.join(trainvaltest_dir, 'images')
            assert os.path.isdir(dst_dir), f"dst dir: {dst_dir} doesn't exist."

            shutil.move(src = src, dst = dst)

    return