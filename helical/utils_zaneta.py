import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


import cv2
from skimage import measure, io, color
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import json
from tqdm import tqdm

zaneta_dir = '/Users/marco/Downloads/zaneta_files'
# zaneta_dir = '/Users/marco/Downloads/example_images_zaneta'
mask_files = glob(os.path.join(zaneta_dir, '*mask.png'))


def _remove_stain_from_names(fold:str):

    old_fp = glob(os.path.join(fold, '*'))
    # new_fp = [os.path.basename(file).split('_', 1) for file in old_fp ]
    # new_fp = [fn[-1] for fn in new_fp ]
    # print(new_fp)
    src2dst = [(file, os.path.join(os.path.dirname(file), os.path.basename(file).split('_', 1)[-1])) for file in old_fp if 'PAS' in file or 'HE' in file]

    for src, dst in tqdm(src2dst, desc='cleaning names'):
        os.rename(src=src, dst=dst)
        # shutil.move(src, dst)



    return




def write_json_files(mask_files:list):
    """ Write json files. """

    # # print(contours.shape)
    # mask = cv2.imread(mask_fp)
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

    def assign_class(color2label:dict, mask_center_rgb:np.ndarray):
        if np.array_equal(color2label['Glo-healthy'], mask_center_rgb): 
            clss = 'Glo-healthy'
        elif np.array_equal(color2label['Glo-unhealthy'], mask_center_rgb):
            clss = 'Glo-unhealthy'
        else: 
            # print(mask_center_rgb)
            raise Exception('Object center does not match any of the object classes, seems to be background instead.')
        return clss
    
    def get_contours(mask_fp:str, min_size:int= 40):

        mask = cv2.imread(mask_fp)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        mask_copy = mask.copy()
        g = cv2.cvtColor(mask_copy,cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(g, 140, 210)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        new_contours = []
        for cont in contours:
            area = cv2.contourArea(cont)
            # print(area)
            if area >= min_size:
                new_contours.append(cont)

        hulls = []
        for i,c in enumerate(new_contours):
            hull = cv2.convexHull(c)
            cv2.drawContours(mask_copy, [hull], 0, (255, 0, 255), 50)
            hulls.append(hull)

        fig, ax = plt.subplots(1, figsize=(12,8))
        plt.imshow(mask_copy)

        return hulls

    def make_json_obj(contours:np.ndarray, mask_fp:np.ndarray):

        mask = cv2.imread(mask_fp)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        color2label = {'Glo-healthy':np.array((0,255,0)), 'Glo-unhealthy':np.array((255,0,0))}
        json_obj = []
        color_rgb =  {"Glo-healthy": -16711936, "Glo-unhealthy": -65536}
        for j, contour in enumerate(contours):
            json_add = {'type':'Feature', 'id':'PathAnnotationObject', 'geometry':{'type':'Polygon', 
                        'coordinates':[[]]}, 'properties':{'classification':{}}}
            contour = np.squeeze(contour)
            xc, yc = 0, 0
            for i, vertex in enumerate(contour): 
                # if i == 0:
                #     print(vertex[1])
                xc += vertex[1]
                yc += vertex[0]
                json_add['geometry']['coordinates'][0].append([int(vertex[0]), (int(vertex[1]))]) # TODO FARE CHECK CHE NON SIA IL CONTRARIO !!
            xc /= len(contour)
            yc /= len(contour)
            xc, yc = int(xc), int(yc)
            clss = assign_class(color2label=color2label, mask_center_rgb=mask[xc, yc])
            if clss is None:
                continue
            json_add['properties']['classification']['name'] = clss
            json_add['properties']['classification']['colorRGB'] = color_rgb[clss]
            json_obj.append(json_add)

        return json_obj
    
    def write_json_obj(mask_fp:str, json_obj:dict):
        assert '_mask.png' in mask_fp
        write_fp = mask_fp.replace('_mask.png', '.json')
        with open(write_fp, 'w') as f:
            json.dump(json_obj, f)
        
        return

    
    for mask_fp in tqdm(mask_files):

        contours = get_contours(mask_fp=mask_fp)
        json_obj = make_json_obj(contours=contours, mask_fp=mask_fp)
        write_json_obj(mask_fp=mask_fp, json_obj=json_obj)
    

    return


write_json_files(mask_files=mask_files)
_remove_stain_from_names(fold=zaneta_dir)

# for fp in image_files:

#     hulls = get_contours(fp=fp)
#     _write_json_files(mask_fp=fp, contours=hulls)









