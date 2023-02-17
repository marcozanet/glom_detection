import json
from typing import List, Tuple
import numpy as np
import os


class JsonLikeParser():

    def __init__(self ,
                fp: str,
                save_folder:str = None,
                # normalize:bool = False,
                label_map:dict = {'Glo-healthy':0, 'Glo-unhealthy':1, 'Glo-NA':2, 'Tissue':3}
                ) -> None:
        
        assert os.path.isfile(fp), ValueError(f"'fp':{fp} is not a valid filepath. ")
        assert os.path.isdir(save_folder) or save_folder is None, ValueError(f"'save_folder':{save_folder} should be either None or a valid filepath. ")
        # assert isinstance(normalize, bool), f"'normalize':{normalize} should be a boolean."
        assert isinstance(label_map, dict), f"'label_map':{label_map} should be a dict."

        self.fp = fp
        # self.normalize = normalize
        self.label_map = label_map
        self.save_folder = save_folder
        if 'gson' in fp:
            print('gson')
            with open(fp, 'rb') as f:
                # text = json.load(file)
                self.text = open(fp,"rb").read()[7:]

        else:
            with open(self.fp, 'r') as f:
                self.text = f.read()


        return

    def _get_masks(self) -> List[Tuple]:
        """ Returns a vertices list (for each object) from a json-like file in txt format (i.e. text file formatted like a json file). """

        # refactor into json style:
        text = self.text.replace('\n', '').replace(' ', '')
        text = [el for el in text.split('"coordinates":[') if '[[' in el]

        # text = ''.join(text)
        # print(text)
        text = [el.split("properties") for el in text]
        text = [el1 for el2 in text for el1 in el2 if "[[" in el1]
        text = [el[:-4] for el in text]
        # text = [el.replace('[', '').replace(']', '') for el in]
        # text = [el for el in text.split('"properties":') if "classification" not in el ]
        # print(text)

        masks = []
        for feature in text: 
            feature = feature.replace('[', '').replace(']', '')
            feature = [int(float(el)) for el in feature.split(',')]
            # feature = [print(i, el) for (i, el) in enumerate(feature) if i == 0]
            # print(feature)
            odds = [el for (i, el) in enumerate(feature) if i%2==1]
            even = [el for (i, el) in enumerate(feature) if i%2==0]
            # print(odds)
            # print(even)
            masks.append(list(zip(even, odds)))
        # print(converted)

        return masks


    def _get_classes(self) -> List[str]:
        """ Returns a class list from a json-like file in txt format (i.e. text file formatted like a json file). """
        
        text = self.text.replace('\n', '').replace(' ', '')
        text = [el for el in text.split('"name":') if "colorRGB" in el]
        text = [el.split('color')[0] for el in text ]
        text = ''.join(text).replace('"', '').split(',')
        classes = [el for el in text if len(el)>0]

        # convert class str to indices:
        classes = [self.label_map[el] for el in classes]

        assert all(isinstance(x, int) for x in classes), f"Not all values in 'classes':{classes} are type int."

        return classes 
    

    def _get_bboxes_from_masks(self, masks: List[Tuple[int]]):
        """ Returns bboxes from masks. """

        bboxes = []
        for mask in masks:
            x = [vertex[0] for vertex in mask]
            y = [vertex[1] for vertex in mask]
            
            x = np.array(x)
            xmin, xmax = x.min(), x.max()
            y = np.array(y)
            ymin, ymax = y.min(), y.max()

            xc, yc = (xmin + xmax)//2, (ymin+ymax)//2
            box_w, box_h = xmax - xmin, ymax-ymin

            bboxes.append((xc, yc, box_w, box_h))

            
        return bboxes
    

    def _write_txt(self, classes:List[str], bboxes:List[tuple]) -> None:
        """ Saves into a file the bbox in YOLO format. NB values are NOT normalized."""

        assert len(classes) == len(bboxes), f"'bboxes' has length: {len(bboxes)}, but 'classes' has length: {len(classes)}"

        rows = list(zip(classes, bboxes ))
        text = ''
        for obj in rows:
            clss = obj[0]
            xc, yc, box_w, box_h = obj[1]
            line = f"{clss}, {xc}, {yc}, {box_w}, {box_h}\n" 
            text += line
        
        fname = os.path.split(self.fp)[1]
        save_fp = os.path.join(self.save_folder, fname) if self.save_folder is not None else self.fp
        with open(save_fp, 'w') as f:
            f.write(text)
        

        return
    
    
    def __call__(self):
        masks = self._get_masks()
        classes = self._get_classes()
        bboxes = self._get_bboxes_from_masks(masks=masks)
        self._write_txt(classes=classes, bboxes=bboxes)
        # self._get_class()
        return


if __name__ == '__main__':
    
    fp = '/Users/marco/Downloads/test_pyramidal/200104066_09_SFOG.mrxs.gson'

    parser = JsonLikeParser(fp = fp, 
                            save_folder='/Users/marco/Documents/labels_txt', 
                            )
    parser()
