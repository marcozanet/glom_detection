import os 
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import geojson


def cropbig_image(crop_coords_file:str):
    crop_coords_file = '/Users/marco/helical_tests/test_manager_segm_muw_sfog/safe_copy_3_slides/200104066_09_SFOG.geojson'

    slide = openslide.OpenSlide(fp)
    region = slide.read_region(location = location , level = self.level, size= (W,H)).convert("RGB")


    with open(crop_coords_file, 'r') as f:
        dictionary = geojson.load(f)
    
    samples = dictionary['features']

    for sample in samples: 
        
        for (x,y) in sample['geometry']['coordinates'][0]:
            print(x,y)
    





    return

cropbig_image(crop_coords_file='a')