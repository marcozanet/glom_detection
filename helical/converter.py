import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from typing import List
import geojson
from glob import glob
import json
# from dirtyparser import JsonLikeParser
from typing import Literal
import numpy as np
from tqdm import tqdm
from loggers import get_logger


class Converter():

    def __init__(self, 
                folder: str, 
                level:int,
                convert_from: Literal['json_wsi_mask', 'jsonliketxt_wsi_mask', 'gson_wsi_mask'], 
                convert_to: Literal['json_wsi_bboxes', 'txt_wsi_bboxes', 'geojson_wsi_mask'],
                save_folder = None,
                map_classes: dict = {'Glo-unhealthy':0, 'Glo-NA':1, 'Glo-healthy':2, 'Tissue':3},
                verbose: bool = False) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""
        
        self.log = get_logger()

        assert os.path.isdir(folder), ValueError(f"'folder': {folder} is not a dir.")
        assert convert_from in ['json_wsi_mask', 'jsonliketxt_wsi_mask', 'gson_wsi_mask'], f"'convert_from'{convert_from} should be in ['json_wsi_mask', 'jsonliketxt_wsi_mask', 'gson_wsi_mask']. '"
        assert convert_to in ['json_wsi_bboxes', 'txt_wsi_bboxes', 'geojson_wsi_mask']
        assert save_folder is None or os.path.isdir(save_folder), ValueError(f"'save_folder':{save_folder} should be either None or a valid dirpath. ")
        assert isinstance(verbose, bool), f"'verbose' should be a boolean."  

        self.convert_from = convert_from
        self.convert_to = convert_to
        self.folder = folder
        self.format_from = convert_from.split('_')[0] if convert_from != 'jsonliketxt_wsi_mask' else 'txt'
        self.format_to = convert_to.split('_')[0]
        self.save_folder = save_folder if save_folder is not None else folder
        self.verbose = verbose
        self.map_classes = map_classes
        self.level = level

        return


    def _get_files(self) -> List[str]:
        """ Collects source files to be converted. """

        
        files = glob(os.path.join(self.folder, f'*.{self.format_from}' ))

        if self.verbose is True:
            self.log.info(f"First 5 files: {files[:5]}", extra={'className': self.__class__.__name__})

        assert len(files)>0, f"No file like {os.path.join(self.folder, f'*.{self.format_from}')} found"

        return files


    def _write_txt(self, data, fp) -> None:
        """ converts data to text and writes it into the same folder. """

        # save path:
        format_from = self.format_from if self.format_from != 'gson' else 'json'
        txt_fp = fp.replace(format_from, self.format_to)
        if self.save_folder is not None:
            fname = os.path.split(txt_fp)[1]
            txt_fp = os.path.join(self.save_folder, fname)
        
        if self.verbose is True:
            self.log.info(f" Writing {txt_fp}", extra={'className': self.__class__.__name__})

        with open(txt_fp, 'w') as f:
            text = ''
            for values in data:
                text += str(values) + '\n'
            text = text.replace('[', '').replace(']', '')
            f.write(text)
            f.close()

        return txt_fp
    

    def _get_dimensions(self, basename:str):
        """ Gets slide dimensions based on the level."""

        slide_fp = basename + ".tif"
        assert os.path.isfile(slide_fp), f"'slide_fp':{slide_fp} is not a valid filepath."
        slide = openslide.OpenSlide(slide_fp)
        W_orig, H_orig = slide.level_dimensions[0]
        W_lev, H_lev = slide.level_dimensions[self.level]

        return W_orig, H_orig,W_lev, H_lev


    

    def _get_bboxes_from_mask(self, fp:str) -> List:
        ''' Gets bboxes values either in .json or in .txt format.
            Output = if 'convert_to' == 'txt_wsi_bboxes':  [class, x_center, y_center, box_w, box_y] (not normalized)
                     if 'convert_to' == 'json_wsi_bboxes':  [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]. '''

        assert os.path.isfile(fp), ValueError(f"Annotation file '{fp}' is not a valid filepath.", extra={'className': self.__class__.__name__})
        
        # read annotation file
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
            try: 
                class_name = glom['properties']['classification']['name']
            except:
                class_name = 'Glo-NA'
            class_value = self.map_classes[class_name]
            
            # saving outer coords (bounding boxes) for each glom
            x_min = 10000000000
            y_min = 10000000000
            x_max = 0
            y_max = 0
            for i, xy in enumerate(vertices[0]):

                # in case of multipolygon:
                if isinstance(xy[0], list):
                    X = np.array([pair[0] for pair in xy])
                    x_min, x_max = X.min(), X.max()
                    # print(x_min, x_max)
                    Y = np.array([pair[1] for pair in xy])
                    y_min, y_max = Y.min(), Y.max()
                    # self.log.info(y_min, y_max)
                
                
                # normal polygon case:
                x = xy[i][0] if isinstance(xy[0], list) else xy[0]
                y = xy[i][1] if isinstance(xy[0], list) else xy[1]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max 
                y_min = y if y < y_min else y_min

            x_c = (x_max + x_min) / 2
            y_c = (y_max + y_min) / 2
            box_w, box_y = x_max - x_min, y_max - y_min

            # NEW: normalize by level dimensions: 
            basename = fp.split('.')[0]
            w_orig, h_orig, w_lev, h_lev = self._get_dimensions(basename=basename)
            x_c, y_c = x_c/w_orig*w_lev, y_c/h_orig*h_lev
            box_w, box_y= box_w/w_orig*w_lev, box_y/h_orig*h_lev

            new_coords.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]) 
            boxes.append([class_value, x_c, y_c, box_w, box_y])

            # self.log.info(f"boxes: {boxes}")

            if self.convert_to == 'txt_wsi_bboxes':
                return_obj = boxes 
            elif self.convert_to == 'json_wsi_bboxes':
                return_obj = new_coords
            else:
                self.log.error(f"Not implemented error", extra={'className': self.__class__.__name__})
                raise NotImplementedError("_get_bboxes function should be used to get either txt bboxes or json wsi bboxes. ")


        return return_obj


    def _convert_json2txt(self, json_file: str) -> str:
        """ Converts .json file with obj annotations on a WSI to .txt format """

        # 1) get bounding boxes values:
        bboxes = self._get_bboxes_from_mask(fp = json_file)
        # 2) save to txt 
        converted_file= self._write_txt(bboxes, fp = json_file)
        self.log.info(f"{self.__class__.__name__}.{'_convert_json2txt'} Converter: ✅ WSi .json annotation converted to WSI .txt annotation. ", extra={'className': self.__class__.__name__})
            
        return converted_file
    
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
    

    def _convert_gson2geojson(self, gson_file:str) -> str:
        """ Converts file in gson format to geojson for visualization purposes. """

        # 1) read in binary and remove extra chars:
        with open(gson_file, encoding='utf-8', errors='ignore', mode = 'r') as f:
            text = f.read()
        text = '[' + text[text.index('{'):]
        json_obj = text.replace("\\", '').replace("/", '')
        # print(text)


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
    
    def _split_multisample_annotation(self, txt_file:str, multisample_loc_file:str) -> None:
        """ Given a WSI txt (not normalised) annotation for samples or ROIs, it splits the annotation file 
            into one file for each sample/ROI within the slide."""
        
        assert os.path.isfile(txt_file), f"'label_file':{txt_file} is not a valid filepath."
        assert os.path.isfile(multisample_loc_file), f"'label_file':{multisample_loc_file} is not a valid filepath."
        assert txt_file.split(".")[-1] == 'txt', f"'txt_file':{txt_file} should have '.txt' format. "

        with open(txt_file, 'r') as f:
            rows = f.readlines()
        
        with open(multisample_loc_file, 'r') as f:
            data = geojson.load(f)
        

        txt_files = []
        txt_fnames = []
        for row in rows:
            clss, xc, yc, box_w, box_h = row.replace(',', '').split(' ')
            clss, xc, yc, box_w, box_h = float(clss), float(xc), float(yc), float(box_w), float(box_h)
            for sample_n, rect in enumerate(data['features']):
                assert len(rect['geometry']['coordinates'][0]) == 5, f"There seems to be more than 4 vertices annotated. "
                save_fp = txt_file.replace('.txt', f"_sample{sample_n}.txt")
                txt_files.append(save_fp)
                txt_fnames.append(f"_sample{sample_n}.txt")
                vertices = rect['geometry']['coordinates'][0][:-1]
                x0, y0 = vertices[0]
                x1, y1 = vertices[2]


                if x0<xc<x1 and y0<yc<y1:
                    xc_new, yc_new = xc-x0, yc-y0 # new ROI coords
                    boxw_new, boxh_new = box_w, box_h # new ROI coords

                    # NEW: normalize by level dimensions:
                    basename = multisample_loc_file.split('.')[0]
                    w_orig, h_orig, w_lev, h_lev = self._get_dimensions(basename=basename)
                    xc_new, yc_new = xc_new/w_orig*w_lev, yc_new/h_orig*h_lev
                    boxw_new, boxh_new = boxw_new/w_orig*w_lev, boxh_new/h_orig*h_lev

                    text = f'{clss}, {xc_new}, {yc_new}, {boxw_new}, {boxh_new}\n'  
                    # save txt file:
                    with open(save_fp, 'a+') as f:
                        f.write(text)

        
        # return a list of txt files for each sample:
        txt_files = list(set(txt_files))
        txt_fnames = list(set(txt_fnames))
        self.log.info(f"{self.__class__.__name__}.{'_split_multisample_annotation'}: ✅ Splitted into {txt_fnames}.")


        return txt_files
    
    def _clear_ROI_objs_(self, txt_file:str):
        assert os.path.isfile(txt_file), f"'txt_file':{txt_file} is not a valid filepath."

        with open(txt_file, 'r') as f:
            lines = f.readlines()
        del_rows = []
        for i, line in enumerate(lines):
            objs = line.replace('\n', '').split(', ')
            h_obj, w_obj = round(float(objs[3])), round(float(objs[4]))
            if h_obj > 5000 or w_obj>5000:
                del_rows.append(i)
        lines = [line for (i, line) in enumerate(lines) if i not in del_rows]
        # print(lines)
        # raise Exception

        
        with open(txt_file, 'w') as f:
            f.writelines(lines)

        return
    
    def _rename_gson2mrxsgson(self):

        files = [os.path.join(self.folder,file) for file in os.listdir(self.folder) if '.gson' in file]
        files = [file.replace('.gson', '.mrxs.gson') for file in files if '.mrxs' not in file ]


        return


    def _convert_jsonliketxt2txt(self, jsonliketxt_file:str) -> None:
        """ Saves into a file the bbox in YOLO format. NB values are NOT normalized."""
        raise NotImplementedError()

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
            self.log.info(f"✅ Already converted: found txt WSI annotation in {self.folder} for {fname}.tiff. Skipping slide.", extra={'className': self.__class__.__name__})
            computed = True
        else:
            computed = False

        return computed
    

    def __call__(self) -> None:
        """ Converts using the proper function depending on the conversion task of choice. """

        # rename all gson to mrxs.gson 
        self._rename_gson2mrxsgson()

        
        files = self._get_files()
        for file in tqdm(files, desc = f"Converting to {self.format_to}"):
            # print(f"⏳ Converting: {os.path.basename(file)}")
            if self.convert_to == 'txt_wsi_bboxes':
                if self.convert_from == 'json_wsi_mask':
                    if self._check_already_converted(file=file):
                        continue
                    converted_file = self._convert_json2txt(json_file = file)
                elif self.convert_from == 'jsonliketxt_wsi_mask':
                    if self._check_already_converted(file=file):
                        continue
                    self._convert_jsonliketxt2txt(jsonliketxt_file=file)
                elif self.convert_from == 'gson_wsi_mask':
                    if self._check_already_converted(file=file):
                        continue
                    converted_file = self._convert_gson2txt(gson_file=file)
            elif self.convert_to == 'geojson_wsi_mask':
                converted_file = self._convert_gson2geojson(gson_file=file)
            # print(f"✅ Converted to: {os.path.basename(converted_file)}")
        self.log.info(f"✅ Converted files saved in: {os.path.dirname(converted_file)}", extra={'className': self.__class__.__name__})
 
        return



        




def test_Converter():
    folder = '/Users/marco/Downloads/train_20feb23/wsi/val/labels'
    converter = Converter(folder = folder, 
                          convert_from='gson_wsi_mask', 
                          convert_to='',
                          save_folder= '/Users/marco/Downloads/heeee', 
                          verbose=False)
    converter()

    return


if __name__ == '__main__':

    test_Converter()