from converter_muw import ConverterMuW
from tiling import Tiler
import os
from tqdm import tqdm
from splitter import Splitter
import shutil


""" Here are reported higher level functions to use inside classes"""

def prepare_muw_data():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\tests\test_TilerConverter\labels'
    save_folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\tests\test_TilerConverter\labels'
    save_root = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else r'D:\marco\datasets\tests\test_TilerConverter\labels'
    level = 2
    show = False
    
    print(" ########################   CONVERTING ANNOTATIONS: ⏳    ########################")
    converter = ConverterMuW(folder = folder, 
                            convert_from='gson_wsi_mask', 
                            convert_to='txt_wsi_bboxes',
                            save_folder= save_folder, 
                            level = level,
                            verbose=False)
    converter()
    print(" ########################   CONVERTING ANNOTATIONS: ✅    ########################")

    print(" ########################    TILING IMAGES: ⏳    ########################")
    tiler = Tiler(folder = folder, 
                  tile_shape= (2048, 2048), 
                  step=512, 
                  save_root= save_root, 
                  level = level,
                  show = show,
                  verbose = True)
    
    target_format = 'tif'
    tiler(target_format=target_format)
    print(" ########################    TILING IMAGES: ✅    ########################")

    print(" ########################    TILING LABELS: ⏳    ########################")
    target_format = 'txt'
    # remove previuos labels if any
    # if target_format == 'txt' and os.path.isdir(os.path.join(save_root, 'labels')):
    #     fold = os.path.join(save_root, 'labels')
    #     files = [os.path.join(fold, file) for file in os.listdir(fold)]
    #     for file in tqdm(files, desc = 'Removing all label files'):
    #         os.remove(file)
        
    tiler(target_format=target_format)
    tiler.test_show_image_labels()
    print(" ########################    TILING LABELS: ✅    ########################")




    return

def split_data():

    print(" ########################    SPLITTING DATA: ⏳     ########################")
    src_dir = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    dst_dir = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    image_format = 'tif'
    ratio = [0.6, 0.2, 0.2]
    task = 'detection'
    verbose = True
    safe_copy = False

    splitter = Splitter(src_dir=src_dir,
                        dst_dir=dst_dir,
                        image_format=image_format,
                        ratio=ratio,
                        task=task,
                        verbose = verbose, 
                        safe_copy = safe_copy)
    splitter()
    # splitter.move_already_tiled(tile_root = '/Users/marco/Downloads/muw_slides')
    # splitter._remove_empty_images()
    print(" ########################    SPLITTING DATA 2: ✅    ########################")

    return




def pipeline(): 

    split_data()

    return

    


if __name__ == '__main__':
    # wsi_folder = '/Users/marco/Downloads/test_folders/test_process_data_and_train/detection/wsi'
    # move_slides_back_from_tiling(wsi_folder=wsi_folder, slide_format='tif')
    # prepare_muw_data()
    pipeline()