import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from glob import glob
from typing import List


class Tiler():

    def __init__(self, folder: str) -> None:
        """ Class for patchification/tiling of WSIs and annotations. """
        return
    

    def _get_files(self, format:str ) -> List[str]:
        """ Collects source files to be converted. """

        files = glob(os.path.join(self.folder, f'*.{format}' ))

        return files

    def _tile_wsi_annotation(fp, shape_patch):
        ''' Patchifies one slide annotation in txt format to patch annotations in txt format. '''

        try:
            # Read WSI:
            slide_fp = fp.replace('_boxes.txt', '.tiff')
            slide = openslide.OpenSlide(slide_fp)
        except:
            print('patchify skipped')
            return

        # Get BB from txt file:
        with open(fp, 'r') as f:
            text = f.readlines()
            f.close()
        
        # for each glom, find corresponding patch:
        for row in text:
            items = row.split(sep = ',')
            xc, yc, box_w, box_h = [float(num) for num in items[1:]]
            w, h = shape_patch, shape_patch
            i = int(xc // w) 
            j = int(yc // h) 
            img_fp = fp.replace('_boxes.txt', f'_{j}_{i}.png') # img that contains the center of that glom

            # convert WSI coords to patch coords:
            xc = xc % w  
            yc = yc % h
            #print(f'Glom patch coords: {xc, yc}')
            
            # write patch annotation txt file:
            txt_fp = img_fp.replace('.png', '.txt')
            text = f'0 {xc/2048} {yc/2048} {box_w/2048} {box_h/2048}\n'

            if os.path.exists(txt_fp):
                mode = 'a' #append if file exists
            else:
                mode = 'w' # write new if does not

            # save txt file:
            with open(txt_fp, mode) as f:
                f.write(text)

                f.close()

        return
    
    def __call__(self, target_format: str) -> None:
        """ Tiles/patchifies WSI or annotations. """

        assert target_format in ['tiff', 'txt'], ValueError(f"Conversion target format should be either 'tiff' or 'txt'. ")
        self._get_files()




        return
