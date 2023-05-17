import os 
from glob import glob
import geojson    

# /Users/marco/Downloads/zaneta_files/detection copia/wsi/train/labels
# /Users/marco/Downloads/zaneta_files/detection/wsi/test/labels/I_4_S_5_ROI_4_PAS.geojson

# TODO FUNZIONA SOLO SU ZANETA, PERCHE' SE CI SONO VARI SAMPLE NON VA
dst_root = '/Users/marco/Downloads/zaneta_files'
path_like = os.path.join(dst_root, 'detection', 'wsi', '*', 'labels', '*.geojson' )
geojson_files = glob(path_like)
# geojson_files = [file for file in geojson_files if os.path.isfile(file)]

for geo_file in geojson_files:
    # self.log.info(f'geojson: {geo_file}')
    print(geo_file)
    delete = False
    with open(geo_file, 'w') as f:
        data_rect = geojson.load(f)
    