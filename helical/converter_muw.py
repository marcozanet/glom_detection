
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
from converter_base import ConverterBase
import time
from cleaner_muw import CleanerMuw


class ConverterMuW(ConverterBase):

    def __init__(self, 
                 *args, 
                 **kwargs) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""
        
        super().__init__(*args, **kwargs)

        return
    
    def _convert_gson2txt(self, gson_file:str) -> str:

        # 1) read in binary and remove extra chars:
        with open(gson_file, encoding='utf-8', errors='ignore', mode = 'r') as f:
            text = f.read()
        
        text = '[' + text[text.index('{'):]
        # 2) convert to json and save:
        gson_file = gson_file.replace('.mrxs', '')
        save_file = gson_file.replace('gson', 'json')
        text = json.loads(text)
        with open(save_file, 'w') as fs:
            json.dump(obj = text, fp = fs)
        self.log.info(f"{self.__class__.__name__}.{'_convert_gson2txt'} Converter: ✅ WSi .json annotation converted to WSI .txt annotation. ", extra={'className': self.__class__.__name__})
        # 3) convert using json converter:
        converted_file = self._convert_json2txt(json_file=save_file)
        # 4) clear txt annotations from ROI objects
        self._clear_ROI_objs_(txt_file=converted_file)

        return  converted_file
    
    def _remove_empty_pair(self, empty_file:str) -> None: 
        """ Removes all files with basename of the passed file from current folder. """

        files = glob(os.path.join(os.path.dirname(empty_file), f"{os.path.basename(empty_file).split('.')[0]}*" ))
        for file in files: 
            os.remove(file)
        self.log.warning(f"{self.class_name}.{'_remove_empty_pair'}: removed {len(files)} files starting with {os.path.basename(empty_file).split('.')}.")

        return
    
    def _check_annotations(self) -> None:
        """ Filters out empty annotations, by removing annotations and corresponding slides. """

        EXT = '.gson'
        files = glob(os.path.join(self.folder, f'*{EXT}'))
        if self.verbose is True:
                self.log.info(f"{self.class_name}.{'_remove_empty_pair'}: checking annotations:")
        
        remaining = 0
        rem_files = []
        for gson_file in files: 
            # 1) check if label is empty:
            with open(gson_file, encoding='utf-8', errors='ignore', mode = 'r') as f:
                text = f.read()
            if '{' not in text: 
                self.log.warning(f"{self.class_name}.{'_convert_gson2geojson'}: removing slide  and its label {gson_file}.")
                del_files = glob(os.path.join(os.path.dirname(gson_file), f"{os.path.basename(gson_file).split('.')[0]}*" ))
                for file in del_files: 
                    os.remove(file)
                self.log.warning(f"{self.class_name}.{'_remove_empty_pair'}: removed {len(del_files)} files starting with {os.path.basename(gson_file).split('.')[0]}.")
            else: 
                remaining += 1
                rem_files.append(gson_file)
        self.log.warning(f"{self.class_name}.{'_remove_empty_pair'}: annotations checked. Remaining files: {remaining}.")

        # for each file verify that also tif exist and at least 1 geojson file: 
        for file in rem_files: 
            geo_file = os.path.join(os.path.dirname(file) , os.path.basename(file).split('.')[0] + f".geojson")
            assert os.path.isfile(geo_file), self.log.error(f"{self.class_name}.{'_check_annotations'}: 'geo_file':{geo_file} doesn't exist. ")
            tif_file = os.path.join(os.path.dirname(file) , os.path.basename(file).split('.')[0] + f".tif")
            assert os.path.isfile(geo_file), self.log.error(f"{self.class_name}.{'_check_annotations'}: 'tif_file':{tif_file} doesn't exist. ")




        return
    

    def _convert_gson2geojson(self, gson_file:str) -> str:
        """ Converts file in gson format to geojson for visualization purposes. """

        # 1) read in binary and remove extra chars:
        with open(gson_file, encoding='utf-8', errors='ignore', mode = 'r') as f:
            text = f.read()
        
        # check if empty:
        assert '{' in text, self.log.error(f"{self.class_name}.{'_convert_gson2geojson'}: label {gson_file} is empty!")
        text = '[' + text[text.index('{'):]
        json_obj = text.replace("\\", '').replace("/", '')

        # convert text to geojson obj:
        json_obj = json.loads(json_obj)
        # print(json_obj)
        # geojson_obj = geojson.dumps(json_obj)

        # save:
        gson_file = gson_file.replace('.mrxs', '')
        geojson_file = gson_file.replace('.gson', '_viz.geojson')
        with open(geojson_file, 'w') as fs:
            geojson.dump(obj = json_obj, fp = fs)

        self.log.info("✅ Converter: WSI .gson annotation converted to WSI .geojson annotation. ", extra={'className': self.__class__.__name__})

        return  geojson_file
    
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

        # 1) rename all files to <basename>_<stain>.<ext>.
        self._rename_all()

        # 2) check annotations empty:
        self._check_annotations()

        # raise NotImplementedError()

        base_names = self._get_files()
        print(len(base_names))
        LEVEL = self.level



        for file in tqdm(base_names, desc = f"Converting annotations for YOLO"):
            
            print(f"file: {file}")
            change_format = lambda fp, fmt: os.path.join(os.path.dirname(fp), os.path.basename(fp).split('.')[0] + f".{fmt}")

            # 1) converting from gson wsi mask to txt wsi bboxes:
            gson_file = file
            self.convert_from = 'gson_wsi_mask'
            self.convert_to = "txt_wsi_bboxes"
            txt_file = change_format(file, 'txt')
            if not os.path.isfile(txt_file):
                self.level = 0 # i.e. output txt is in original coordinates
                self._convert_gson2txt(gson_file=gson_file) # this will also save an intermediate json file
            assert os.path.isfile(txt_file), self.log.error(f"{self.class_name}.{'__call__'}: txt-converted file {txt_file} doesn't exist.")

            # 2) splitting the geojson file into multiple txt files (1 for sample/ROI):
            geojson_file = change_format(gson_file, 'geojson')
            self.convert_from = 'geojson_wsi_mask'
            self.convert_to = "txt_wsi_bboxes"
            # time.sleep(5)
            # print(txt_file)
            # print(txt_file.replace('.txt', '_sample0.txt'))   
            if not os.path.isfile(txt_file.replace('.txt', '_sample0.txt')): # already computed some samples for this slide, skipping
                self.level = LEVEL
                self._split_multisample_annotation(txt_file=txt_file, multisample_loc_file=geojson_file) # this is now in level coordinates


            # cleaning





 
        return


        




def test_Converter():
    folder = '/Users/marco/converted_march'
    save_folder = '/Users/marco/converted_march'
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