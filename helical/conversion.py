
import os
from typing import List
import geojson
from glob import glob
import json
from dirtyparser import JsonLikeParser
from typing import Literal


class Converter():

    def __init__(self, 
                folder: str, 
                convert_from: Literal['json_wsi_mask', 'jsonliketxt_wsi_mask'], 
                convert_to: Literal['json_wsi_bboxes', 'txt_wsi_bboxes'],
                save_folder = None,
                verbose: bool = False) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""

        assert os.path.isdir(folder), ValueError(f"'folder': {folder} is not a dir.")
        assert convert_from in ['json_wsi_mask', 'jsonliketxt_wsi_mask'], f"'convert_from'{convert_from} should be in ['json_wsi_mask', 'jsonliketxt_wsi_mask']. '"
        assert convert_to in ['json_wsi_bboxes', 'txt_wsi_bboxes']
        assert save_folder is None or os.path.isdir(save_folder), ValueError(f"'save_folder':{save_folder} should be either None or a valid dirpath. ")
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."  

        self.convert_from = convert_from
        self.convert_to = convert_to
        self.folder = folder
        self.format_from = convert_from.split('_')[0] if convert_from != 'jsonliketxt_wsi_mask' else 'txt'
        self.format_to = convert_to.split('_')[0]
        self.save_folder = save_folder
        self.verbose = verbose


    def _get_files(self) -> List[str]:
        """ Collects source files to be converted. """

        
        files = glob(os.path.join(self.folder, f'*.{self.format_from}' ))

        if self.verbose is True:
            print(files)

        assert len(files)>0, f"No file like {os.path.join(self.folder, f'*.{self.format_from}')} found"

        return files


    def _write_txt(self, data, fp) -> None:
        """ converts data to text and writes it into the same folder. """

        # save path:
        txt_fp = fp.replace('.json', '.txt')
        if self.save_folder is not None:
            fname = os.path.split(txt_fp)[1]
            txt_fp = os.path.join(self.save_folder, fname)

        with open(txt_fp, 'w') as f:
            text = ''
            for values in data:
                text += str(values) + '\n'
            text = text.replace('[', '').replace(']', '')
            f.write(text)
            f.close()

        return
    

    def _get_bboxes_from_mask(self, fp:str) -> List:
        ''' Gets bboxes values either in .json or in .txt format.
            Output = if 'convert_to' == 'txt_wsi_bboxes':  [class, x_center, y_center, box_w, box_y] (not normalized)
                     if 'convert_to' == 'json_wsi_bboxes':  [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]. '''

        assert os.path.isfile(fp), ValueError(f"Annotation file '{fp}' is not a valid filepath.")
        
        # read annotation file
        print('opening file')
        print(fp)
        with open(fp, 'r') as f:
            try:
                data = geojson.load(f)
            except:
                raise Exception
            print(data)
            
        # saving outer coords (bboxes) for each glom:
        new_coords = []
        boxes = []
        x_min = 10000000000
        y_min = 10000000000
        x_max = 0
        y_max = 0

        # access polygon vertices of each glom
        for glom in data:
            vertices = glom['geometry']['coordinates']
            
            # saving outer coords (bounding boxes) for each glom
            x_min = 10000000000
            y_min = 10000000000
            x_max = 0
            y_max = 0
            for _, xy in enumerate(vertices[0]):
                x = xy[0]
                y = xy[1]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max 
                y_min = y if y < y_min else y_min


            x_c = round((x_max + x_min) / 2, 2)
            y_c = round((y_max + y_min) / 2, 2)  
            box_w, box_y = round((x_max - x_min), 2) , round((y_max - y_min), 2)
            new_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]) 
            boxes.append([0, x_c, y_c, box_w, box_y])

            if self.convert_to == 'txt_wsi_bboxes':
                return_obj = boxes 
            elif self.convert_to == 'json_wsi_bboxes':
                return_obj = new_coords
            else:
                raise NotImplementedError("_get_bboxes function should be used to get either txt bboxes or json wsi bboxes. ")


        return return_obj


    def _convert_json2txt(self, json_file: str) -> None:
        """ Converts .json file with obj annotations on a WSI to .txt format """

        # 1) get bounding boxes values:
        bboxes = self._get_bboxes_from_mask(fp = json_file)
        # 2) save to txt 
        self._write_txt(bboxes, fp = json_file)

        print("Converter: WSi .json annotation converted to WSI .txt annotation. ")
            
        return
    
    def _convert_gson2txt(self, gson_file:str):
        file = "/Users/marco/Downloads/test_pyramidal/200104066_09_SFOG.mrxs.gson"
        with open(file, 'rb') as f:
            # text = json.load(file)
            text = open(file,"rb").read()[7:]


        return
    
    # def _converttxt2txt(self, txt_file:str) -> None:
    #     """ Converts .txt file with mask annotations on a WSI to bboxes annotations in .txt format """
    #     # 1) get bounding boxes values:
    #     bboxes = self._get_bboxes_from_mask(fp = txt_file)
    #     # 2) save to txt 
    #     self._write_txt(bboxes, fp = txt_file)
    #     print("Converter: WSi .json annotation converted to WSI .txt annotation. ")
    #     return

    def _convert_jsonliketxt2txt(self, jsonliketxt_file:str) -> None:
        """ Saves into a file the bbox in YOLO format. NB values are NOT normalized."""

        parser = JsonLikeParser(fp = jsonliketxt_file, 
                                save_folder=self.save_folder, 
                                label_map = {'Glo-healthy':0, 'Glo-unhealthy':1, 'Glo-NA':2, 'Tissue':3})
        parser()

        return
    


    def _check_already_converted(self, file: str) -> bool:
        """ Checks whether conversion has already been computed. """

        fname = os.path.split(file)[1].split(f'.{self.format_from}')[0]

        files = glob(os.path.join(self.save_folder, f'*.{self.format_to}'))
        files = [file for file in files if fname in file]

        if len(files) > 0:
            print(f"Converter: found txt WSI annotation in {self.folder} for {fname}.tiff. Skipping slide.")
            computed = True
        else:
            computed = False

        return computed
    
    def __call__(self) -> None:
        """ Converts using the proper function depending on the conversion task of choice. """

        files = self._get_files()
        for file in files:
            if self.convert_from == 'json_wsi_mask' and self.convert_to == 'txt_wsi_bboxes':
                if self._check_already_converted(file=file):
                    continue
                self._convert_json2txt(json_file = file)
            elif self.convert_from == 'jsonliketxt_wsi_mask' and self.convert_to == 'txt_wsi_bboxes':
                self._convert_jsonliketxt2txt(jsonliketxt_file=file)
 

        return


        






def test_Converter():
    folder = '/Users/marco/Downloads/test_pyramidal'
    converter = Converter(folder = folder, 
                          convert_from='jsonliketxt_wsi_mask', 
                          convert_to='txt_wsi_bboxes',
                          save_folder= '/Users/marco/Downloads/test_pyramidal', 
                          verbose=True)
    converter()

    return


if __name__ == '__main__':

    test_Converter()