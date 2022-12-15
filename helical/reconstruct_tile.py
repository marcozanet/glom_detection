import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_path_folders(coords_folder, crops_folder, reconstructed_tiles_folder, wsi_fn):
    """  Sets paths and folders.  """

    crops_fns = [file for file in os.listdir(crops_folder) if 'png' in file and wsi_fn in file]
    coords_fns = [file.replace('png', 'txt') for file in crops_fns]
    crops_fps = [os.path.join(crops_folder, file) for file in crops_fns]
    coords_fps = [os.path.join(coords_folder, file) for file in coords_fns]
    tiles_fns = set([file.split('glom')[0] for file in coords_fns])
    # create save folder 
    if not os.path.isdir(reconstructed_tiles_folder):
        os.makedirs(reconstructed_tiles_folder)
    
    return tiles_fns, crops_fps, coords_fps



def reconstruct_tiles(tiles_fns, crops_fps, coords_fps, reconstructed_tiles_folder:str, save = True, ):
    """  Given tiles names, crops filepaths and coordinates filepaths, it reconstruct the original tile.  """
    
    for tile_fn in tqdm(tiles_fns, desc = 'Reconstructing tiles'):

        # combine coords of crops and crops images to reassable tile:
        TILE_SHAPE = (2048, 2048)
        tile = np.zeros(shape = TILE_SHAPE)
        crops_in_file_fps = [file for file in crops_fps if tile_fn in file]
        coords_in_file_fps = [file for file in coords_fps if tile_fn in file]

        #  assign each crop to its location within the tile
        for crop_fp, coords_fp in zip(crops_in_file_fps, coords_in_file_fps):

            # get coords:
            with open(coords_fp, 'r') as f:
                text = f.read()
                f.close()
            y_min, y_max, x_min, x_max  = np.array([int(num) for num in text.split(',') ])
            # get crop:
            crop = io.imread(crop_fp)
            # copy paste crop in the right location:

            tile[ y_min: y_max, x_min:x_max ] = np.where( crop > 125, crop, tile[y_min: y_max, x_min:x_max])

        # save tile
        if save is True:
            tile = np.uint8(np.where(tile > 125, 255, 0))
            save_fp = os.path.join(reconstructed_tiles_folder, tile_fn + '.png')
            io.imsave( save_fp, tile, check_contrast= False  )

    return 


def reconstruct_WSI(coords_folder, crops_folder, reconstructed_tiles_folder, wsi_fn, plot = False):
    """"  Given crops, their coords within the tile and tiles, reconstructs the original WSI.  """

    # TODO: PER ORA FUNZIONA CON UNA SOLA SLIDE MA DEVE ANDARE PER TANTE SLIDE 
    # set paths and folders
    tiles_fns, crops_fps, coords_fps = set_path_folders(coords_folder, crops_folder, reconstructed_tiles_folder, wsi_fn)

    # reconstruct tiles
    reconstruct_tiles(tiles_fns, crops_fps, coords_fps, reconstructed_tiles_folder=reconstructed_tiles_folder)
    
    # get wsi names of test 
    wsi_fns = list(set([file.split('_')[0] for file in tiles_fns]))
    print(f'Slides to be reconstructed: ')
    print(*wsi_fns)

    # get wsi dims (max indxs):
    i_max, j_max = 0, 0
    for tile_fn in tiles_fns:
        idxs = tile_fn.split('_')[1:3]
        I, J = int(idxs[0]), int(idxs[1])
        i_max = I if I > i_max else i_max
        j_max = J if J > j_max else j_max

    # read each tile and fill the WSI with tiles:
    wsi = np.zeros(shape = (i_max * 2048, j_max * 2048))
    print(f'WSI shape: {wsi.shape}')
    for tile_fn in tqdm(tiles_fns, desc= 'Reconstructing WSI'):
        tile_fp = os.path.join( reconstructed_tiles_folder , tile_fn + '.png')
        tile = io.imread(tile_fp)
        idxs = tile_fn.split('_')[1:3]
        I, J = int(idxs[0]), int(idxs[1])
        nail = wsi[ (I-1) * 2048 : (I * 2048),  (J-1) * 2048 : (J * 2048) ]
        wsi[ ((I-1) * 2048) : (I * 2048),   ((J-1) * 2048) : (J * 2048) ] = np.where( tile > 125, tile, nail)

    # plot 
    if plot is True:
        fig = plt.figure()
        plt.imshow(wsi)
        plt.show()
        fig.savefig(fname = f'plot_{ wsi_fns[0]}')
    io.imsave(fname = wsi_fns[0], arr=wsi)


    # save prediction on the whole slide:
    # TODO

    return



def reconstruct(preds_folder: str,
                coords_folder: str,
                crops_folder: str, 
                reconstructed_tiles_folder: str, 
                plot = True
                
                ):
    """ Reconstructs all WSIs contained in the data_folder. """

    # get fnames of the wsis to be reconstructed:
    wsis_fns = list(set([file.split('_')[0] for file in os.listdir(preds_folder) if 'DS' not in file and 'png' in file]))

    for wsi_fn in wsis_fns:
        print(f'Reconstructing {wsi_fn}:')
        reconstruct_WSI(coords_folder, crops_folder, reconstructed_tiles_folder, wsi_fn, plot )

    return

if __name__ == '__main__':

    coords_folder = '/Users/marco/hubmap/unet_data/reconstruct'
    crops_folder = '/Users/marco/hubmap/data/preds'
    reconstructed_tiles_folder = '/Users/marco/hubmap/unet_data/reconstructed'
    # reconstruct_WSI(coords_folder, crops_folder, reconstructed_tiles_folder, plot = True) 
    preds_folder = '/Users/marco/hubmap/unet_data/test/preds'
    reconstruct(preds_folder=preds_folder,
                coords_folder = coords_folder,
                crops_folder = crops_folder,
                reconstructed_tiles_folder= reconstructed_tiles_folder,
                plot = True )