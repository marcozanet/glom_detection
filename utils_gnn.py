import os 
from glob import glob
from typing import List


def _get_txt_files(folder: str) -> dict:
    """ Collects txt files for each slide and puts slides and their list of txt files into a dictionary. """

    files = glob(os.path.join(folder, '*.txt'))
    # print(len(files))
    # print(files[0])
    wsi_fnames = [os.path.split(file)[1] for file in files]
    wsi_fnames = list(set(name.split('_')[0] for name in wsi_fnames))
    # print(len(wsi_fnames))
    # print(wsi_fnames)
    
    paired_files = []
    for slide in wsi_fnames:
        # print(slide)
        txt_files = [file for file in files if slide in file]
        paired_files.append(txt_files)
    
    assert len(paired_files) == len(wsi_fnames)

    txt_files = dict(zip(wsi_fnames, paired_files))

    return txt_files

def _get_xy_txt( W: int, H:int ):
    """ """



    return


def test_get_txt_files():

    folder = '/Users/marco/hubmap/training/train/model_train/labels'
    _get_txt_files(folder=folder)

    return


if __name__ == '__main__':
    test_get_txt_files()