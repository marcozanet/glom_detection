
from typing import Literal, Tuple
import os
from loggers import get_logger
from decorators import log_start_finish
from converter_muw import ConverterMuW
from splitter import Splitter



class ProcessorManager(): 
    def __init__(self,
                src_root: str, 
                dst_root: str, 
                slide_format: Literal['tif', 'tiff'],
                label_format: Literal['gson', 'mrxs.gson'],
                tiling_shape: Tuple[int],
                tiling_step: int,
                split_ratio = [0.7, 0.15, 0.15], 
                task = Literal['detection', 'segmentation', 'both'],
                safe_copy: bool = False,
                verbose: bool = False,
                empty_perc: float = 0.1) -> None:
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.slide_format = slide_format
        self.label_format = label_format
        self.tiling_shape = tiling_shape
        self.tiling_step = tiling_step
        self.split_ratio = split_ratio
        self.task = task
        self.verbose = verbose
        self.safe_copy = safe_copy
        

        self.log = get_logger()

        return

    def _split_data(self):

        print(" ########################    SPLITTING DATA: ⏳     ########################")

        @log_start_finish(class_name=self.__class__.__name__, func_name='_split_data', msg= f" Splitting slides")
        def do():
            splitter = Splitter(src_dir=self.src_root,
                                dst_dir=self.dst_root,
                                image_format=self.slide_format,
                                ratio=self.split_ratio,
                                task=self.task,
                                verbose = self.verbose, 
                                safe_copy = self.safe_copy)
            splitter()
            return
        
        do()
        # splitter.move_already_tiled(tile_root = '/Users/marco/Downloads/muw_slides')
        # splitter._remove_empty_images()
        print(" ########################    SPLITTING DATA 2: ✅    ########################")

        return
    

    def __call__(self) -> None:

        self._split_data()

        return


    # def prepare_muw_data(self):

        
    #     print(" ########################   CONVERTING ANNOTATIONS: ⏳    ########################")
    #     converter = ConverterMuW(folder = folder, 
    #                             convert_from='gson_wsi_mask', 
    #                             convert_to='txt_wsi_bboxes',
    #                             save_folder= save_folder, 
    #                             level = level,
    #                             verbose=False)
    #     converter()
    #     print(" ########################   CONVERTING ANNOTATIONS: ✅    ########################")

    #     print(" ########################    TILING IMAGES: ⏳    ########################")
    #     tiler = Tiler(folder = folder, 
    #                 tile_shape= (2048, 2048), 
    #                 step=512, 
    #                 save_root= save_root, 
    #                 level = level,
    #                 show = show,
    #                 verbose = True)
        
    #     target_format = 'tif'
    #     tiler(target_format=target_format)
    #     print(" ########################    TILING IMAGES: ✅    ########################")

    #     print(" ########################    TILING LABELS: ⏳    ########################")
    #     target_format = 'txt'
    #     # remove previuos labels if any
    #     # if target_format == 'txt' and os.path.isdir(os.path.join(save_root, 'labels')):
    #     #     fold = os.path.join(save_root, 'labels')
    #     #     files = [os.path.join(fold, file) for file in os.listdir(fold)]
    #     #     for file in tqdm(files, desc = 'Removing all label files'):
    #     #         os.remove(file)
            
    #     tiler(target_format=target_format)
    #     tiler.test_show_image_labels()
    #     print(" ########################    TILING LABELS: ✅    ########################")




    #     return
    



def test_ProcessorManager(): 

    # import sys 
    # system = 'mac' if sys.platform == 'darwin' else 'windows'
    # folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    # save_folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    # save_root = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    # level = 2
    # show = False

    # CONFIG
    src_root = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    dst_root = '/Users/marco/Downloads/test_folders/test_process_data_and_train'
    slide_format = 'tif'
    label_format = 'gson'
    split_ratio = [0.6, 0.2, 0.2]
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (2048,2048)
    tiling_step = 512

    manager = ProcessorManager(src_root=src_root,
                               dst_root=dst_root,
                               slide_format=slide_format,
                               label_format=label_format,
                               split_ratio=split_ratio,
                               tiling_shape=tiling_shape,
                               tiling_step=tiling_step,
                               task=task,
                               verbose=verbose,
                               safe_copy=safe_copy)
    manager()


    return


if __name__ == '__main__': 
    test_ProcessorManager()