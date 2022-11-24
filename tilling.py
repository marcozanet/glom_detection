
import os 
import geojson
from typing import List

class Tiler():


    def __init__(self) -> None:

        pass


    def get_bboxes(W:int, H:int, fp:str) -> List:
        ''' Converts .json segmentation annotations to bboxes in txt YOLO format '''
        
        # read json file
        with open(fp, 'r') as f:
            data = geojson.load(f)
            
        # saving outer coords (bboxes) for each glom:
        gloms = 0
        boxes = []
        x_min = 10000000000
        y_min = 10000000000
        x_max = 0
        y_max = 0

        # access polygon vertices of each glom
        for glom in data:
            gloms += 1
            vertices = glom['geometry']['coordinates']
            
            # saving outer coords (bounding boxes) for each glom
            x_min = 10000000000
            y_min = 10000000000
            x_max = 0
            y_max = 0
            for i, xy in enumerate(vertices[0]):
                x = xy[0]
                y = xy[1]
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max 
                y_min = y if y < y_min else y_min

            if x_max > W:
                raise Exception()
            if y_max > H:
                raise Exception()

            x_c =  (x_max + x_min) / 2 
            y_c = (y_max + y_min) / 2  
            box_w, box_y = (x_max - x_min) , (y_max - y_min)
            boxes.append([0, x_c, y_c, box_w, box_y])


        return boxes