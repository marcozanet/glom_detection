from converter_muw import ConverterMuW
from tiling import Tiler
import os
from tqdm import tqdm
""" Here are reported higher level functions to use inside classes"""

def prepare_muw_data():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    folder = '/Users/marco/Downloads/test_folders/test_tiler/test_manyslide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\val\labels'
    save_folder = '/Users/marco/Downloads/test_folders/test_tiler/test_manyslide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\val\labels'
    level = 2
    save_root = '/Users/marco/Downloads/test_folders/test_tiler/test_manyslide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\val\labels'
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


if __name__ == '__main__':
    prepare_muw_data()