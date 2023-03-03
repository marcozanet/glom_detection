from converter_muw import ConverterMuW
from tiling import Tiler
import os
from tqdm import tqdm
""" Here are reported higher level functions to use inside classes"""

def prepare_muw_data():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    save_folder = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    save_root = '/Users/marco/Downloads/test_folders/test_tiler/test_1slide' if system == 'mac' else  r'D:\marco\datasets\muw_retiled\wsi\test\labels'
    level = 2
    show = False
    
    print(" ########################   CONVERTING ANNOTATIONS: ⏳    ########################")
    converter = ConverterMuW(folder = folder, 
                            convert_from='gson_wsi_mask', 
                            convert_to='txt_wsi_bboxes',
                            save_folder= save_folder, 
                            level = level,
                            verbose=False)
    converter()
    print(" ########################   CONVERTING ANNOTATIONS: ✅    ########################")

    print(" ########################    TILING IMAGES: ⏳    ########################")
    tiler = Tiler(folder = folder, 
                  tile_shape= (2048, 2048), 
                  step=512, 
                  save_root= save_root, 
                  level = level,
                  show = show,
                  verbose = True)
    
    target_format = 'tif'
    tiler(target_format=target_format)
    print(" ########################    TILING IMAGES: ✅    ########################")

    print(" ########################    TILING LABELS: ⏳    ########################")
    target_format = 'txt'
    # remove previuos labels if any
    # if target_format == 'txt' and os.path.isdir(os.path.join(save_root, 'labels')):
    #     fold = os.path.join(save_root, 'labels')
    #     files = [os.path.join(fold, file) for file in os.listdir(fold)]
    #     for file in tqdm(files, desc = 'Removing all label files'):
    #         os.remove(file)
        
    tiler(target_format=target_format)
    tiler.test_show_image_labels()
    print(" ########################    TILING LABELS: ✅    ########################")




    return

    
    def _get_n_tiles(self,
                     files: list, 
                     overlapping: bool = False,
                     save_folder: str = None) -> None:
        """ Returns the number of x and y tiles that are computed by patchifying the file. """

        class_name = self.__class__.__name__
        func_name = '_get_n_tiles'
        files = [fp.replace(self.format, 'tif') for fp in files]
        assert all([os.path.isfile(fp) for fp in files]), ValueError(f"some path in {files} is not a valid filepath.")
        save_folder = os.path.join(self.save_root, 'images') if save_folder is None else save_folder

        @log_start_finish(class_name=class_name, func_name=func_name, msg = f" Getting n_tiles:" )
        def do():        
            
            n_tiles = {}
            for fp in files:
                w, h = self.tile_shape

                # 1) read slide:
                try:
                    slide = openslide.OpenSlide(fp)
                except:
                    self.log.error(f"{class_name}.{func_name}: ❌ Couldn t open file: '{os.path.basename(fp)}'. Skipping." )
                    continue
                W, H = slide.dimensions

                # 2) if file has multi_samples -> region = sample:
                if self.multiple_samples is True:
                    # get file with location of image/label samples within the slide:
                    multisample_loc_file = self._get_multisample_loc_file(fp, file_format='geojson', mode='labels')
                    sample_locations = self._get_location_w_h(fp = multisample_loc_file) if multisample_loc_file is not None else [{'location':(0,0), 'w':W, 'h':H}]
                else:
                    multisample_loc_file = None
                    sample_locations = [{'location':(0,0), 'w':W, 'h':H}]


                for sample_n, sample in enumerate(tqdm(sample_locations, desc= "Samples")):
                    
                    location, W, H = sample['location'], sample['w'], sample['h']
                    
                    # 1) reading region:
                    self.log.info(f"{class_name}.{func_name}: ⏳ Reading region ({W, H}) of sample_{sample_n}:")
                    try:
                        region = slide.read_region(location = location , level = self.level, size= (W,H)).convert("RGB")
                    except:
                        self.log.error(f"{class_name}.{func_name}: ❌ Reading region failed")

                    # 2) converting to numpy array:
                    self.log.info(f"{class_name}.{func_name}: ⏳ Converting to numpy sample_{sample_n}:")
                    try:
                        np_slide = np.array(region)
                    except:
                        self.log.error(f"{class_name}.{func_name}: ❌ Conversion to numpy.")

                    # 3) patchification:
                    self.log.info(f"{class_name}.{func_name}: ⏳ Patchifying sample_{sample_n}:")
                    try:
                        if overlapping is False:
                            patches = patchify(np_slide, (w, h, 3), step =  self.step )
                            sample_fname = os.path.basename(fp).split('.')[0] + f"_sample{sample_n}"
                            n_tiles[sample_fname] = (patches.shape[0], patches.shape[1])
                        else:
                            raise NotImplementedError()
                    except:
                        self.log.error(f"{class_name}.{func_name}: ❌ Patchifying.")
                    self.log.info(f"{class_name}.{func_name}: n_tiles = {n_tiles}")
            return n_tiles
        
        ret_obj = do()

        return ret_obj


if __name__ == '__main__':
    prepare_muw_data()