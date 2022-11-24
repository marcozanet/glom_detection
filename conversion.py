
import os
from typing import List
import geojson
from glob import glob


class Converter():

    def __init__(self, 
                folder: str, 
                convert_from: str, 
                convert_to:str ) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""

        assert os.path.isdir(folder), ValueError(f"'folder': {folder} is not a dir.")
        assert convert_from in ['json_wsi_mask']
        assert convert_to in ['json_wsi_bboxes', 'txt_wsi_bboxes']


        self.convert_from = convert_from
        self.convert_to = convert_to
        self.folder = folder
        self.format = convert_from.split('_')[0]


    def _get_files(self) -> List[str]:
        """ Collects source files to be converted. """

        format = self.convert_from.split('_')[0]
        files = glob(os.path.join(self.folder, f'*.{format}' ))

        return files


    def _write_txt(self, data, fp):
        """ converts data to text and writes it into fp by replacing json ~~. """

        with open(fp.replace('.json', '_boxes.txt').replace('hub_bb', 'bb'), 'w') as f:
            text = ''
            for values in data:
                text += str(values) + '\n'
            text = text.replace('[', '').replace(']', '')
            f.write(text)
            f.close()

        return


    def _get_bboxes_from_mask(self, fp:str) -> List:
        ''' Gets bboxes values either in .json or in .txt format.
            Output = if 'convert_to' == 'txt_wsi_bboxes':  [class, x_center, y_center, box_w, box_y]. 
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
            # print(glom)
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


    def _convert_json2txt(self, json_file: str):
        """ Converts .json file with obj annotations on a WSI to .txt format """

        # 1) get bounding boxes values:
        bboxes = self._get_bboxes_from_mask(fp = json_file)
        print(bboxes)
        self._write_txt(bboxes, fp = json_file)
            
        return
    
    def __call__(self):
        """ Converts using the proper function depending on the conversion task of choice. """

        files = self._get_files()
        for file in files:
            if self.convert_from == 'json_wsi_mask' and self.convert_to == 'txt_wsi_bboxes':
                self._convert_json2txt(json_file = file)

        return
        


def test_Converter():
    """ Tests converter class. """
    
    folder = '/Users/marco/Downloads/new_source'
    converter = Converter(folder = folder, convert_from='json_wsi_mask', convert_to='txt_wsi_bboxes' )
    converter()


    return

if __name__ == '__main__':

    test_Converter()