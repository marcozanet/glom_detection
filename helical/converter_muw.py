
import os
from typing import List
import geojson
from glob import glob
import json
# from dirtyparser import JsonLikeParser
from typing import Literal
import numpy as np
from tqdm import tqdm
from loggers import get_logger
from converter import Converter
import time


class ConverterMuW(Converter):

    def __init__(self, 
                 *args, 
                 **kwargs) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""
        
        super().__init__(*args, **kwargs)

        return
    

    
    def _get_files(self) -> List[str]:
        """ Collects source files to be converted. """

        
        gson_files = glob(os.path.join(self.folder, f'*.gson'))
        geojson_files = glob(os.path.join(self.folder, f'*.geojson'))


        if self.verbose is True:
            self.log.info(f"gson_files:{len(gson_files)}, geojson files:{len(geojson_files)}", extra={'className': self.__class__.__name__})

        assert len(gson_files)>0, f"No gson files found. "
        assert len(geojson_files)>0, f"No geojson_files files found. "

        base_names = list(set([file.split('.')[0] for file in gson_files]))
        base_names.extend(list(set([file.split('.')[0] for file in gson_files])))
        base_names = list(set(base_names))
        # filtering slides that don't have either a gson file or a geojson file
        base_names = [basename for basename in base_names if basename+'.mrxs.gson' in gson_files and basename+'.geojson' in geojson_files ]

        return base_names
    
    
    def _check_already_converted(self, file: str) -> bool:
        """ Checks whether conversion has already been computed. """

        fname = os.path.split(file)[1].split(f'.{self.format_from}')[0]
        files = glob(os.path.join(self.save_folder, f'*.{self.format_to}'))
        files = [file for file in files if fname in file]

        if len(files) > 0:
            self.log.info(f"✅ Already converted: found txt WSI annotation in {self.folder} for {fname}.tiff. Skipping slide.", extra={'className': self.__class__.__name__})
            computed = True
        else:
            computed = False

        return computed
    
    def __call__(self) -> None:
        """ Converts using the proper function depending on the conversion task of choice. """

        base_names = self._get_files()
        LEVEL = self.level

        for file in tqdm(base_names, desc = f"Converting annotations for YOLO"):
            
            # 1) converting from gson wsi mask to txt wsi bboxes:
            gson_file = file+'.mrxs.gson'
            self.convert_from = 'gson_wsi_mask'
            self.convert_to = "txt_wsi_bboxes"
            txt_file = file+'.txt'
            if not os.path.isfile(txt_file):
                self.level = 0 # i.e. output txt is in original coordinates
                self._convert_gson2txt(gson_file=gson_file) # this will also save an intermediate json file

            # 2) splitting the geojson file into multiple txt files (1 for sample/ROI):
            geojson_file = file+'.geojson'
            self.convert_from = 'geojson_wsi_mask'
            self.convert_to = "txt_wsi_bboxes"
            # time.sleep(5)
            # print(txt_file)
            # print(txt_file.replace('.txt', '_sample0.txt'))   
            if not os.path.isfile(txt_file.replace('.txt', '_sample0.txt')): # already computed some samples for this slide, skipping
                self.level = LEVEL
                self._split_multisample_annotation(txt_file=txt_file, multisample_loc_file=geojson_file) # this is now in level coordinates

            # changing back the txt file to




 
        return


        




def test_Converter():
    folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide'
    save_folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide'
    level = 2
    converter = ConverterMuW(folder = folder, 
                            convert_from='gson_wsi_mask', 
                            convert_to='txt_wsi_bboxes',
                            save_folder= save_folder, 
                            level = level,
                            verbose=False)
    converter()
    

    return


if __name__ == '__main__':

    test_Converter()