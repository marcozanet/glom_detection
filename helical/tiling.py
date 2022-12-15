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
import warnings
from patchify import patchify
import numpy as np 
from PIL import Image
from tqdm import tqdm


class Tiler():

    def __init__(self, 
                folder: str, 
                tile_shape: tuple = (2048, 2048),
                save_root = None) -> None:
        """ Class for patchification/tiling of WSIs and annotations. """

        assert os.path.isdir(folder), ValueError(f"Provided 'folder':{folder} is not a valid dirpath.")
        assert isinstance(tile_shape, tuple), TypeError(f"'tile_shape' should be a tuple of int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape' should be a tuple of int.")
        assert save_root is None or os.path.isdir(save_root), ValueError(f"'save_root':{save_root} should be either None or a valid dirpath. ")

        self.folder = folder 
        self.tile_shape = tile_shape
        self.save_root = save_root
        
        return
    

    def _get_tile_txt_annotations(self, fp: str, save_folder: str = None):
        ''' Makes tile txt annotations in YOLO format (normalized) out of (not normalized) txt annotations for the entire image.
            Annotations tiles are of shape 'tile_shape' and are only made around each object contained in the WSI annotation, since YOLO doesn't 
            need annotations for empty images. 
            fp = path to WSI (not normalized) annotation in .txt format '''

        assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")
        save_folder = os.path.join(self.save_root, 'labels') if save_folder is None else save_folder

        # Get BB from txt file:
        with open(fp, 'r') as f:
            text = f.readlines()
            f.close()
        
        # for each glom, find corresponding patch:
        for row in text:
            # get values:
            items = row.split(sep = ',')
            xc, yc, box_w, box_h = [float(num) for num in items[1:]]
            clss = items[0]
            w, h = self.tile_shape[0], self.tile_shape[1]
            # get the position of the tile in WSI coords
            i = int(xc // w) 
            j = int(yc // h) 
            # normalize coords for YOLO:
            xc = xc % w  
            yc = yc % h
            # save
            save_fp = fp.replace('.txt', f'_{j}_{i}.txt') # img that contains the center of that glom
            if save_folder is not None:
                fname = os.path.split(save_fp)[1]
                save_fp = os.path.join(save_folder, fname)
            self._write_txt(clss, xc, yc, box_w, box_h, save_fp)

        print(f"Tiler: WSI .txt annotation tiled into .txt annotation tiles, saved in {save_folder}. ")

        return

    
    def _get_tile_images(self, 
                        fp: str, 
                        tile_shape: tuple = (2048, 2048), 
                        overlapping: bool = False,
                        save_folder: str = None) -> None:
        """ Tiles the WSI into tiles and saves them into the save_folder. """
        
        assert os.path.isfile(fp), ValueError(f"{fp} is not a valid filepath.")
        assert isinstance(tile_shape, tuple) and len(tile_shape) == 2, TypeError(f"'tile_shape':{tile_shape} should be a tuple of two int.")
        assert isinstance(tile_shape[0], int) and isinstance(tile_shape[1], int), TypeError(f"'tile_shape':{tile_shape} should be a tuple of two int.")
        assert isinstance(overlapping, bool), TypeError(f"'overlapping' should be a boolean. ")
        save_folder = os.path.join(self.save_root, 'images') if save_folder is None else save_folder

        w, h = tile_shape
        fname = os.path.split(fp)[1]

        # 1) read slide:
        try:
            slide = openslide.OpenSlide(fp)
        except:
            warnings.warn(f'Couldn t open file: {fp}. Skipping. ')
            return
        W, H = slide.dimensions

        # 2) convert to numpy:
        slide = slide.read_region(location = (0,0), level = 0, size= (W,H)).convert("RGB")
        slide = np.array(slide)
        if overlapping is False:
            patches = patchify(slide, (w, h, 3), step = w )
        else:
            raise NotImplementedError()
        
        # 3) save patches:
        patches = patches[:, :, 0, ...]
        for i in tqdm(range(patches.shape[0]), desc= f"Tiling '{fname}'"):
            for j in range(patches.shape[1]):
                save_fp = fp.replace('.tiff',f'_{i}_{j}.png')
                if save_folder is not None:
                    fname = os.path.split(save_fp)[1]
                    save_fp = os.path.join(save_folder, fname)
                pil_img = Image.fromarray(patches[i, j])
                pil_img.save(save_fp)

        print(f"Tiler: WSI tiled into .png tiles, saved in {save_folder}. ")

        return

    
    def __call__(self, target_format: str, save_folder: str = None) -> None:
        """ Tiles/patchifies WSI or annotations. """

        assert target_format in ['tiff', 'txt'], ValueError(f"Patchification target format should be either an image in 'tiff' format or an annotation in 'txt' format. ")
        assert save_folder is None or os.path.isdir(save_folder), ValueError(f"'save_folder':{save_folder} should be either None or a valid dirpath. ")
        
        default_folder = 'images' if target_format == 'tiff' else 'labels'
        save_folder = os.path.join(self.save_root, default_folder) if save_folder is None else save_folder


        # 1) make save folders:
        os.makedirs(save_folder, exist_ok=True)

        # 2) get WSI/annotations:
        files = self._get_files(format = target_format)

        # 3) tile files:
        for file in files:
            
            fname = os.path.split(file)[1].split('.')[0]

            # check if already tiled:
            if self._check_already_computed(fname, target_format, save_folder= save_folder):
                continue
            
            if target_format == 'txt':
                self._get_tile_txt_annotations(fp = file, save_folder=save_folder)
            else:
                self._get_tile_images(fp = file, save_folder=save_folder )




        return


    ############ HELPER FUNCTIONS ############

    def _write_txt(self, clss, xc, yc, box_w, box_h, save_fp):

        # write patch annotation txt file:
        w = self.tile_shape[0]
        h = self.tile_shape[1]
        text = f'{clss} {xc/w} {yc/h} {box_w/w} {box_h/h}\n'

        # save txt file:
        with open(save_fp, 'a+') as f:
            f.write(text)
        
        return


    def _get_files(self, format:str ) -> List[str]:
        """ Collects source files to be converted. """

        files = glob(os.path.join(self.folder, f'*.{format}' ))

        # sanity check:
        already_patched = glob(os.path.join(self.folder, f'*_?_?.{format}' ))
        if len(already_patched) > 0:
            print(f"Tiler: Warning: found tile annotations (e.g. {already_patched[0]}) in source folder. ")
        
        files = [file for file in files if file not in already_patched]

        return files
    

    def _check_already_computed(self, fname: str, format: str, save_folder:str ):
        """ Checks if tiling is already computed for this WSI; if so, skips the slide. 
            Hypothesis: tiling is considered to be done if at least 2 tiles are found in 'save_folder'. """

        format = 'png' if format == 'tiff' else format
        files = glob(os.path.join(save_folder, f'*.{format}'))
        files = [file for file in files if fname in file ]
        computed = True if len(files) > 2 else False
        if computed:
            print(f"Tiler: found .{format} tiles in '{save_folder}' for {fname}.tiff. Skipping slide.")

        return computed
    




def test_Tiler():

    folder = '/Users/marco/Downloads/new_source'
    tile_shape = (4096, 4096)
    tiler = Tiler(folder = folder, tile_shape= tile_shape, save_root='/Users/marco/Downloads/folder_random')
    tiler(target_format='txt')
    tiler(target_format='tiff')


    return




if __name__ == '__main__':

    test_Tiler()