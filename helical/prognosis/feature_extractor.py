import os
from glob import glob
import numpy as np



class FeatureExtractor():

    def __init__(self,
                folder:str,
                confidence:bool = False) -> None:

        assert os.path.isdir(folder), f"'folder':{folder} is not a valid dirpath. "
        assert isinstance(confidence, bool), f"'confidence':{confidence} should be boolean."

        self.folder = folder 
        self.confidence = confidence

        return


    def _get_list_gloms(self) -> dict:
        """ Given the prediction folder, collects a list of healthy/sclerosed 
            glom for each biopsy in the folder and assembles biopsy names and list of glom classes in a dictionary. """

        # 1) get label files 
        files = glob(os.path.join(self.folder, '*.txt'))
        assert len(files) > 0, f"No files found in folder. "

        # get unique names of biopsies
        wsi_names = list(set([os.path.basename(file).split('_')[0] for file in files]))

        # initialize dictionary for biopsy/glomclasses
        wsi_classes = {}
        for name in wsi_names:
            wsi_classes[name] = []

        # save all glom classes from labels in respective biopsy
        class_idx = 0 if self.confidence is False else -1
        for file in files:
            with open(file, 'r') as f:
                text = f.readlines()
            glom_classes = [line.split(' ')[class_idx] for line in text if line.split(' ')[0].isnumeric()]
            wsi_name = os.path.basename(file).split('_')[0]
            wsi_classes[wsi_name].extend(glom_classes)
        
        assert len(wsi_classes) > 0, f"'wsi_classes' dictionary is empty. "
        
        return  wsi_classes


    def get_aggregator(self, aggregator:str = 'mean') -> dict:
        """ Given a dict of biopsy and glom classes, saves the percentage of sclerosed gloms for each biopsy in a dict. """
        
        assert aggregator in ['mean', 'median', 'sum', 'product'], f"'aggregator' should be one of ['mean', 'median', 'sum', 'product']."

        # 1) get all classes:
        wsi_classes = self._get_list_gloms()

        # 2) aggregate classes:
        wsi_aggregated = {}
        for key, value in wsi_classes.items():
            value = np.array([int(num) for num in value])
            if aggregator == 'mean':
                aggregated = np.mean(value)
            elif aggregator == 'median':
                aggregated = np.median(value)
            elif aggregator == 'sum':
                aggregated = np.sum(value)
            elif aggregator == 'product':
                aggregated = np.prod(value)
            wsi_aggregated[key] = aggregated

        assert len(wsi_aggregated) > 0, f"'wsi_classes' dictionary is empty. "

        return wsi_aggregated





if __name__ == "__main__":
    folder = '/Users/marco/datasets/muw_exps/detection/train/labels'
    extractor = FeatureExtractor(folder=folder, 
                                 confidence=False)
    feature = extractor.get_aggregator(aggregator='mean')
    print(feature)
