
import os
from typing import List
import geojson
from glob import glob


class Converter():

    def __init__(self, 
                folder: str, 
                convert_from: str, 
                convert_to:str,
                save_folder = None) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""

        assert os.path.isdir(folder), ValueError(f"'folder': {folder} is not a dir.")
        assert convert_from in ['json_wsi_mask']
        assert convert_to in ['json_wsi_bboxes', 'txt_wsi_bboxes']
        assert save_folder is None or os.path.isdir(save_folder), ValueError(f"'save_folder':{save_folder} should be either None or a valid dirpath. ")

        self.convert_from = convert_from
        self.convert_to = convert_to
        self.folder = folder
        self.format = convert_from.split('_')[0]
        self.save_folder = save_folder


    def _get_files(self) -> List[str]:
        """ Collects source files to be converted. """

        format = self.convert_from.split('_')[0]
        files = glob(os.path.join(self.folder, f'*.{format}' ))

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

        assert os.path.isfile(fp), ValueError(f"Geojson annotation file '{fp}' is not a valid filepath.")
        
        # read json file
        with open(fp, 'r') as f:
            data = geojson.load(f)
            
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
    
    
    def __call__(self) -> None:
        """ Converts using the proper function depending on the conversion task of choice. """

        files = self._get_files()
        for file in files:
            if self.convert_from == 'json_wsi_mask' and self.convert_to == 'txt_wsi_bboxes':
                if self._check_already_converted(json_file=file):
                    continue
                self._convert_json2txt(json_file = file)

        return


    def _check_already_converted(self, json_file: str) -> bool:
        """ Checks whether conversion has already been computed. """

        fname = os.path.split(json_file)[1].split('.json')[0]

        files = glob(os.path.join(self.folder, '*.txt'))
        files = [file for file in files if fname in file]

        if len(files) > 0:
            print(f"Converter: found txt WSI annotation in {self.folder} for {fname}.tiff. Skipping slide.")
            computed = True
        else:
            computed = False

        return computed
        






def test_Converter():
    folder = '/Users/marco/Downloads/new_source'
    converter = Converter(folder = folder, 
                          convert_from='json_wsi_mask', 
                          convert_to='txt_wsi_bboxes',
                          save_folder= '/Users/marco/Downloads/folder_random' )
    converter()

    return


if __name__ == '__main__':

    test_Converter()