
import os
from typing import List
import geojson
from glob import glob
import numpy as np
import json
# from dirtyparser import JsonLikeParser
from typing import Literal
import numpy as np
from tqdm import tqdm
from loggers import get_logger
from converter_base import ConverterBase
import time
from cleaner_muw import CleanerMuw
import os 
from PIL import Image, ImageDraw
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

class Converter(ConverterBase):

    def __init__(self, 
                 *args, 
                 **kwargs) -> None:
        """ Offers conversion capabilities from/to a variety of formats. 
            json_wsi_mask"""
        
        super().__init__(*args, **kwargs)

        return
    
    

    def _split_multisample_json(self, json_wsi_file:str, geojson_sample_rect:str): 
        """ If file is multisample, splits the annotation json wsi 
            into multiple json sample annotation files."""
        
        assert os.path.isfile(json_wsi_file), f"'json_wsi_file':{json_wsi_file} is not a valid filepath."
        assert os.path.isfile(geojson_sample_rect), f"'geojson_sample_rect':{geojson_sample_rect} is not a valid filepath."

        # read geojson rectangle file
        with open(geojson_sample_rect, 'r') as f:
            data_rect = geojson.load(f)
        n_samples = len(data_rect['features'])
        
        # read json annotation file
        with open(json_wsi_file, 'r') as f:
            data_annotations = json.load(f)

        # loop through annotation vertices of every object
        def write_sample_file(json_wsi_file, sample_n, json_obj): 

            fp = json_wsi_file.replace('.json', f"_sample{sample_n}.json")
            assert 'json' in fp, self.log.error(f"{fp} doesn't contain 'json'.")
            with open(fp, 'w') as f:
                json.dump(obj=json_obj, fp=f)
            # self.log.info(f"saved {fp}")
            return

        
        # slide_fp = geo
        

        # sample_json_dict = {}
        for j, sample in enumerate(data_rect['features']):
            
            if j > 9:
                raise NotImplementedError() # there will be issues later in tiler (wildcard [0-9] wouldn't work)

            sample_json_dict = {'features':[]}
            rect_vertices = sample['geometry']['coordinates'][0]
            assert len(rect_vertices) == 5, f"Geojson sample has {len(rect_vertices)} vertices, but should have 5."
            x_start,y_start = rect_vertices[0]
            x_end,y_end = rect_vertices[2]
            assert x_start!=x_end and y_start!=y_end, f"Some sample rectangle vertices are coincident: x_start:{x_start}, x_end:{x_end}, y_start:{y_start}, y_end:{y_end}"
            self.log.info(f"Rectangle for {os.path.basename(json_wsi_file)}: x_start:{x_start}, x_end:{x_end}, y_start:{y_start}, y_end:{y_end}")
            
            if len(data_annotations)==0:
                self.log.error(f"data annotations is empty: {data_annotations}. Making empty label. ")
                write_sample_file(json_wsi_file=json_wsi_file, sample_n=0, json_obj={})
                self.log.error(f"done write sample file")


            for i, feat in enumerate(data_annotations):
                # compute center: 
                xc = np.array([tup[1] for tup in feat['geometry']['coordinates'][0]]).mean()
                yc = np.array([tup[0] for tup in feat['geometry']['coordinates'][0]]).mean() # TODO CHECK THAT X E Y SONO NEL CORRETTO ORDINE
                
                if self.data_source == 'muw':
                    xc, yc = yc, xc 

                # if center outside sample rectangle, continue:
                if not x_start<=xc<=x_end:
                    self.log.error(f'xc non cade dentro i vertici del rettangolo :{x_start}<={xc}<={x_end}. Skipping glom.')
                    continue 
                if not y_start<=yc<=y_end:
                    self.log.error(f'yc non cade dentro i vertici del rettangolo:{y_start}<={yc}<={y_end}. Skipping glom.')
                    continue 
                # if obj is Tissue and not glom, skip: 
                try:
                    _class = feat['properties']['classification']['name']
                except: 
                    _class = "Glo-NA"

                if _class == 'Tissue':
                    if self.data_source == 'zaneta':
                        self.log.error("_class == 'Tissue', there shouldn't be any Tissue class in 'zaneta' annotations.")
                    continue
                # print(f"center: {xc,yc}")
                # else write the vertices of the obj:
                # sample_json_dict[j]['features'].append({'coordinates': []})
                coords = []
                file_coords = feat['geometry']['coordinates'][0]
                try: 
                    _ = [ (xi,yi) for xi, yi in feat['geometry']['coordinates'][0] ]
                except:
                    self.log.warn(f"File: '{json_wsi_file}' at row: '{feat['geometry']['coordinates'][0]}' would raise Exception. \nTaking away one more parenthesis than usual." )
                    file_coords = feat['geometry']['coordinates'][0][0]



                # try:
                for xi, yi in file_coords:
                    # clip x and y: e.g. if xi exceeds x_min or x_max, take x_min or x_max as new xi
                    if self.data_source == 'muw':
                        clip_x, clip_y = xi, yi
                        xy = (clip_x - x_start, clip_y - y_start)
                    elif self.data_source == 'zaneta':
                        clip_x, clip_y = (min(max(x_start, xi), x_end), min(max(y_start, yi), y_end) )
                        xy = (clip_x - x_start, clip_y - y_start)
                        xy = (clip_x, clip_y)
                    else:
                        raise NotImplementedError()
                    # if clipped_xy != (xi,yi):
                        # print(f"xi,yi: {xi,yi}")
                        # print(f"xi,yi clipped: {clipped_xy}")
                        # print(clipped_xy)
                        # raise NotImplementedError()
                    coords.append(xy)
                sample_json_dict['features'].append( {'coordinates':coords,'classification':_class}  )
                # except: 
                #     print(json_wsi_file)
                #     print(feat['geometry']['coordinates'][0])
                #     raise NotImplementedError()

                
                # print(f'splitting writing {json_wsi_file}')
                write_sample_file(json_wsi_file=json_wsi_file, sample_n=j, json_obj=sample_json_dict)

                

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
    

    def _convert_muw(self, file:str, LEVEL):

        print(f"CALLING CONVERTER muw: {file}")

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
        if not os.path.isfile(txt_file.replace('.txt', '_sample0.txt')): # already computed some samples for this slide, skipping
            self.level = LEVEL
            self._split_multisample_annotation(txt_file=txt_file, multisample_loc_file=geojson_file) # this is now in level coordinates
        
        # if level then all geojson need to be downsampled :
        # if LEVEL != 0:
        json_file = change_format(gson_file, 'json')
        self.log.info(json_file)
        self._split_multisample_json(json_wsi_file=json_file, geojson_sample_rect=geojson_file)

        self._downscale_geojson_file(json_file=json_file)
        self._interpolate_vertices(json_file=json_file)

        # self.log.info(f'tesssstinnnnnngggg muw: {file}')
        # self._test_show_label_image(file=file.split('.')[0])

        return

    def _convert_hubmap(self, file:str, LEVEL:int):

        self.log.info(f"CALLING CONVERTER hubmap: {file}")

        change_format = lambda fp, fmt: os.path.join(os.path.dirname(fp), os.path.basename(fp).split('.')[0] + f".{fmt}")

        geojson_file = change_format(file, 'geojson')
        json_file = change_format(file, 'json')
        # print('wooooo')
        self.log.info(f'Converting with converter hubmap: {file}')
        # raise NotImplementedError()
        self._split_multisample_json(json_wsi_file=json_file, geojson_sample_rect=geojson_file)
        # self._downscale_geojson_file(json_file=json_file)
        self._interpolate_vertices(json_file=json_file)

        self.log.info(f'tesssstinnnnnngggg hubmap: {file}')
        # self._test_show_label_image(file=file.split('.')[0])

        return
    
    def __call__(self) -> None:
        """ Converts using the proper function depending on the conversion task of choice. """
        
        # 1) rename all files to <basename>_<stain>.<ext>.
        self._rename_all()
        # 2) check annotations empty:
        self._check_annotations()
        base_names = self._get_files()
        print(len(base_names))
        LEVEL = self.level
            
        for file in tqdm(base_names, desc = f"Converting annotations for YOLO"):
            print(file)
            if self.data_source == 'muw':
                self._convert_muw(file=file, LEVEL=LEVEL)
            elif self.data_source == 'hubmap':
                print('converting hubmap')
                self._convert_hubmap(file=file, LEVEL=LEVEL)
            elif self.data_source == 'zaneta':
                self.log.info('converting with convert_hubmapppp')
                self._convert_hubmap(file=file, LEVEL=0)
            else: 
                raise NotImplementedError()
            
 
        return
    
    
    def _interpolate_vertices(self, json_file: str, times:int = 3):
        
        basename = os.path.basename(json_file).split('.json')[0]
        json_sample_files = [file for file in glob(os.path.join(os.path.dirname(json_file), f"{basename}*.json")) if 'sample' in file]
        assert len(json_sample_files), self.log.error(f"'json_sample_files':{json_sample_files} is empty.")

        def interpolate(vertices: list, times:int): 
            for _ in range(times):
                X = [x for x,y in vertices]
                Y = [y for x,y in vertices]
                X_new, Y_new = [], []
                for i in range(len(X)-1):
                    x0, x1 = X[i], X[i+1]
                    y0, y1 = Y[i], Y[i+1]
                    x_new = int((x0+x1)/2)
                    y_new = int((y0+y1)/2)
                    X_new.extend((x0,x_new))
                    Y_new.extend((y0,y_new))
                X_new.append(X[-1])
                Y_new.append(Y[-1])
                new_vertices = list(zip(X_new, Y_new))
                vertices = new_vertices

            return new_vertices

        for json_sample in json_sample_files:
            
            # read/copy
            with open(json_sample, 'r') as f:
                data = json.load(f)

            if len(data)==0:
                self.log.warn(f"File is empty. Skipping vertices interpolation")
                continue
            


            data_new = data.copy()

            # loop through gloms/vertices and interpolate:
            for sample_n, gloms in enumerate(data_new['features']):
                vertices = gloms['coordinates']
                new_vertices = interpolate(vertices=vertices, times=times)
                data_new['features'][sample_n]['coordinates'] = new_vertices

            # save new labels:
            with open(json_sample, 'w') as f:
                json.dump(obj = data_new, fp = f)

        return
    
    
    def _downscale_geojson_file(self, json_file:str):
        """ """
        basename = os.path.basename(json_file).split('.json')[0]
        json_sample_files = [file for file in glob(os.path.join(os.path.dirname(json_file), f"{basename}*.json")) if 'sample' in file]
        print(f'downsampling {json_sample_files}')
        assert len(json_sample_files), f"'json_sample_files':{json_sample_files} is empty."

        # get level dims and original dims to rescale:
        slide_file = json_file.split('.')[0] + '.tif'
        slide = openslide.OpenSlide(slide_file)
        w_0, h_0 = slide.dimensions
        w_lev, h_lev = slide.level_dimensions[self.level]

        # read json samples and write new rescaled json sample files:
        for json_sample in json_sample_files:
            
            # read/copy
            with open(json_sample, 'r') as f:
                data = json.load(f)

            if len(data)==0:
                self.log.warn(f"File is empty. Skipping downscale geojson file.")
                continue
            data_new = data.copy()


            for sample_n, gloms in enumerate(data_new['features']):
                vertices = gloms['coordinates']
                if self.data_source == 'zaneta':
                    new_vertices = [ [int(x/w_0*w_lev), int(y/h_0*h_lev)] for x,y in vertices]
                elif self.data_source == 'muw':
                    new_vertices = [ [int(x/w_0*w_lev), int(y/h_0*h_lev)] for y,x in vertices]
                data_new['features'][sample_n]['coordinates'] = new_vertices

            # new_geojson = geojson_file.replace('.geojson', 'copy.geojson')
            # print(data_new)
            # data_new = json.dumps(data_new)
            with open(json_sample, 'w') as f:
                json.dump(obj = data_new, fp = f)
            print(f'saved {json_sample}')

            # raise NotImplementedError()


        return



        




def test_Converter():
    folder = '/Users/marco/converted_march'
    save_folder = '/Users/marco/converted_march'
    level = 2
    converter = Converter(folder = folder, 
                            convert_from='gson_wsi_mask', 
                            convert_to='txt_wsi_bboxes',
                            save_folder= save_folder, 
                            level = level,
                            verbose=False)
    converter()
    

    return


if __name__ == '__main__':

    test_Converter()