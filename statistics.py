import os 
from skimage import io 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List



def plot_labels(root: str)-> None:
    """ Looks into e.g. train, test, val sets and plots statistics on labels. """

    assert isinstance(root, str), TypeError(f"'root' should be str.")
    assert os.path.isdir(root), ValueError(f"'root':{root} is not a valid dir.")

    def _get_files(root: str, format:str = 'png') -> List[str]:
        """ Helper functiion to collect files from root and subfolders. """

        collected_files = []
        for root, _, files in os.walk(root):
            files_found = [os.path.join(root, file) for file in files if f".{format}" in file and os.path.isfile(os.path.join(root,file))]
            if isinstance(files_found, list):
                collected_files.extend(files_found)
            elif isinstance(files_found, str):
                collected_files.append(files_found)

        return collected_files

    # 1) collect txt files:
    txt_files = _get_files(root = root, format= 'txt' )
    
    # get image size from an image 
    image = io.imread(txt_files[0].replace('.txt', '.png').replace('labels', 'images'))
    W, H, C = image.shape
    print(f"Image size: {W, H}")

    # 2) read files and labels:
    txt_df = []
    for file in txt_files:
        with open(file, 'r') as f:
            text = f.read()
        if len(text) == 0: # if file empty
            continue
        text = text.split('\n')
        text = [string.split(' ') for string in text if len(string)>0 ]
        txt_df.extend(text)
    
    # 3) collect background images:
    images = _get_files(root = root, format = 'png')
    n_bg_images = len(images) - len(txt_files)

    # 4) create df
    cols = [ 'Class','xc', 'yc', 'w', 'h']
    types = {'Class': 'int', 'xc':'float', 'yc':'float', 'w':'float', 'h':'float'}
    df = pd.DataFrame(txt_df, columns = cols)
    df = df.astype(types)
    df['Class'] = df['Class'] + 1
    add_rows = np.zeros(shape = (n_bg_images, len(df.columns)))
    df_empty = pd.DataFrame(add_rows, columns = cols)
    df_empty = df_empty.astype(types)
    df = pd.concat(objs = (df, df_empty))

    # unnormalize:
    df['xc'] = df['xc'] * W
    df['yc'] = df['yc'] * H
    df['w'] = df['w'] * W
    df['h'] = df['h'] * H
    df['Area'] = df['w'] * df['h']
    df = df.astype({'Class': 'int', 'xc':'int', 'yc':'int', 'w':'int', 'h':'int', 'Area': 'int'})
    
    # 5) plot:
    print(df.head())
    data = sns.countplot(data = df, x = 'Class')

    return


def test_plot_labels():

    plot_labels(root = '/Users/marco/datasets/muw_exps/detection/train')

    return



if __name__ == '__main__':
    test_plot_labels()