import os 
from glob import glob

def get_missing_converted(wsi_folder:str):
    """ Given a root folder with a list of wsi files, it checks that all slides 
        have a label, a json ROI file and a txt file. """
    
    wsis = glob(os.path.join(wsi_folder, '*.tif'))
    missing_txt = []
    missing_geojgson = [] 
    for fp in wsis: 
        label1 = fp.replace('.tif', '.gson')
        label2 = fp.replace('.tif', '.mrxs.gson')
        if os.path.isfile(label1):
            label = os.path.basename(label1)
        elif os.path.isfile(label2):
            label = os.path.basename(label2)
        else:
            print(f"❌ fp:{os.path.basename(fp)} has no label.")
        # txt label:
        txt_label = fp.replace('.tif', '.txt')
        if not os.path.isfile(txt_label):
            print(f"❌ wsi:{os.path.basename(fp)} doesn't have a txt label. Convert using Converter(convert_from='gson_wsi_mask').")
            missing_txt.append(fp)
        # geojson roi:
        geojson_label = fp.replace('.tif', '.geojson')
        if not os.path.isfile(geojson_label):
            print(f"❌ wsi: {os.path.basename(fp)} doesn't have a geojson label.")
            missing_geojgson.append(fp)            
        # txt sample label:
        txt_samples = glob(os.path.join(wsi_folder, f"{os.path.basename(fp.split('.')[0])}*sample*.txt"))
        # print(os.path.join(wsi_folder, f"{os.path.basename(fp)}*sample*.txt"))
        if len(txt_samples)<1:
            print(f"❌ wsi: {os.path.basename(fp)} doesn't have a txt sample label. \n     Convert ROI annotations using Converter(convert_from='gson_wsi_mask').")
            missing_geojgson.append(fp)   
    
    



    return

def test_get_missing_converted(): 
    wsi_folder = '/Users/marco/Downloads/muw_slides'
    get_missing_converted(wsi_folder=wsi_folder)
    return


if __name__ == '__main__':
    test_get_missing_converted()
