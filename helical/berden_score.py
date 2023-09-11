import os 
from glob import glob
from configurator import Configurator
from utils import get_config_params
from typing import Literal

class BerdenScoreComputer(Configurator):

    def __init__(self, config_yaml_fp:str) -> None:
        super().__init__()
        self.params = get_config_params(config_yaml_fp,'berden_score_computer')
        self.config_yaml_fp = config_yaml_fp
        self._set_all_attributes()
        return
    
    def _set_all_attributes(self)->None:
        # func_n = self._set_all_attributes.__name__
        self.class_name = self.__class__.__name__
        self.slides_dir = self.params['slides_dir']
        self.cnn_exp_dir = self.params['cnn_exp_dir']
        self.slides_fmt = self.params['slides_fmt']
        self.crops_fmt = self.params['crops_fmt']
        self._parse_args()
        self._edit_attrs()

        
        return
    
    def _edit_attrs(self)->None:
        func_n = self._edit_attrs.__name__
        self.cnn_exp_dir = os.path.join(self.cnn_exp_dir, 'infer')
        assert os.path.isdir(self.cnn_exp_dir), self.assert_log(f"'cnn_exp_dir':{self.cnn_exp_dir} is not a valid dirpath.", func_n=func_n)
        return
    
    def _parse_args(self)->None:
        """ Parses class arguments. """
        func_n = self._parse_args.__name__
        assert os.path.isdir(self.slides_dir), self.assert_log(f"'slides_dir':{self.slides_dir} is not a valid dirpath.", func_n=func_n)
        assert os.path.isdir(self.cnn_exp_dir), self.assert_log(f"'cnn_exp_dir':{self.cnn_exp_dir} is not a valid dirpath.", func_n=func_n)
        assert os.path.basename(self.cnn_exp_dir) != 'infer', self.assert_log(f"'cnn_exp_dir':{self.cnn_exp_dir} should point to cnn exp root dir, not to a specific mode dir.", func_n=func_n)
        
        return
    
    def _get_last_exp(self)->str:
        """ Gets last exp dir from exp dir folder. """
        func_n = self._get_last_exp.__name__
        subfolds = glob(os.path.join(self.cnn_exp_dir, '*'))
        subfolds = [fold for fold in subfolds if ".DS" not in os.path.basename(fold)]
        assert len(subfolds)>0, self.assert_log(f"No exp found in {self.cnn_exp_dir}.", func_n=func_n)
        last_exp = max(subfolds, key=os.path.getctime)
        self.crops_dir = last_exp
        return
    
    def get_unique_slides_from_folder(self)->list:
        """ Gets unique slides from slides folder. """
        func_n = self.get_unique_slides_from_folder.__name__
        slides = list(set(glob(os.path.join(self.slides_dir, f'*.{self.slides_fmt}'))))
        assert len(slides)>0, self.assert_log(f"No 'slides' found in {self.slides_dir} with {self.slides_fmt} fmt", func_n=func_n)
        self.unique_slides = slides
        return 
    
    def get_crops_classes(self)->dict:
        """ Gets for each basename crop its classification, excluding FP. """
        # 1) get crops from dir 
        func_n = self.get_crops_classes.__name__
        classes = glob(os.path.join(self.crops_dir, '*'))
        classes = [class_dir for class_dir in classes if os.path.basename(class_dir)!='false_positives' and 'DS' not in os.path.basename(class_dir)]
        crops = list(set(glob(os.path.join(self.crops_dir, '*', f'*.{self.crops_fmt}'))))
        crops = [file for file in crops if os.path.dirname(file) in classes]
        assert len(crops)>0, self.assert_log(f"No 'crops' found in {self.crops_dir} with {self.crops_fmt} fmt", func_n=func_n)
        self.d_crop_class = {os.path.basename(file).split('.')[-2]:os.path.basename(os.path.dirname(file)) for file in crops}
        self.classes = [os.path.basename(_class) for _class in classes]
        self.crops = crops
        return 
    
    def _count_sample_classes(self):
        samples_fn = list(set(['_'.join(os.path.basename(crop_fn).split('_')[:3]) for crop_fn in self.crops]))
        get_sample_fn = lambda crop_fn: '_'.join(crop_fn.split('_')[:3])
        d_sample_classes= {sample_fn:{_class:0 for _class in self.classes} for sample_fn in samples_fn }
        for crop_fn, _class in self.d_crop_class.items():
            sample = get_sample_fn(crop_fn=crop_fn)
            d_sample_classes[sample][_class] += 1
        self.d_sample_classes = d_sample_classes
        self.samples_fn = samples_fn
        return
    
    def _count_slide_classes(self):
        slides_fn = list(set(['_'.join(os.path.basename(crop_fn).split('_')[:2]) for crop_fn in self.crops]))
        get_slide_fn = lambda crop_fn: '_'.join(crop_fn.split('_')[:2])
        d_slide_classes = {slide_fn:{_class:0 for _class in self.classes} for slide_fn in slides_fn }
        for crop_fn, _class in self.d_crop_class.items():
            slide = get_slide_fn(crop_fn=crop_fn)
            d_slide_classes[slide][_class] += 1
        self.d_slide_classes = d_slide_classes
        self.slides_fn = slides_fn
        return
    
    def _count_tot_gloms(self, d_group_classes:dict)->dict:
        d_group_totgloms = {}
        for group, classes in d_group_classes.items():
            tot_group_gloms = sum(classes.values())
            d_group_totgloms[group] = tot_group_gloms
        self.d_group_totgloms = d_group_totgloms
        return  
    

    def _compute_perc_gloms(self, d_group_classes:dict):
        d_group_percentages = d_group_classes.copy()
        for group, classes in d_group_classes.items():
            for clss, ngloms in classes.items():
                d_group_percentages[group][clss] = round((ngloms/self.d_group_totgloms[group]), 4)
        self.d_group_percentages = d_group_percentages
        return
    
    def _compute_berden(self, fns:list ):
        """ Helper func to compute berden score via its definition. """
        
        d_group_berden = {group:None for group in fns}
        for group, classes in self.d_group_percentages.items():
            sclerosis_perc = 0
            for clss, perc in classes.items():
                if perc>0.5:
                    if clss=='Glomerulus':
                        d_group_berden[group]='Focal'
                    elif clss=='Cellular Crescent':
                        d_group_berden[group]='Crescentic'
                    else:
                        d_group_berden[group]='Sclerotic'
                else:
                    sclerosis_perc+= perc
            if d_group_berden[group] is None:
                if sclerosis_perc>0.5:
                    d_group_berden[group]='Sclerotic'
                else:
                    d_group_berden[group]='Mixed'
        self.d_group_berden=d_group_berden
        return
    

    def compute_berden_score(self, groupby:Literal['slides','samples']):
        """ Computes berden score grouped either by slides or samples. """
        func_n = self.compute_berden_score.__name__

        # 1) Compute class percentages:
        if groupby=='slides':
            self._count_tot_gloms(d_group_classes=self.d_slide_classes)
            self._compute_perc_gloms(d_group_classes=self.d_slide_classes)
        elif groupby=='samples':
            self._count_tot_gloms(d_group_classes=self.d_sample_classes)
            self._compute_perc_gloms(d_group_classes=self.d_sample_classes)
        else: 
            raise ValueError(f"'groupby':{groupby} should be one of ['slides','samples']")
        self.format_msg(f"✅ Class percentages for {groupby} computed. Class percentages: {self.d_group_percentages}", func_n=func_n)
        
        # 2) Compute Berden Score from %
        fns = self.slides_fn if groupby=='slides' else self.samples_fn
        self._compute_berden(fns=fns)
        self.format_msg(f"✅ Berden scores for {groupby} computed. Berden output classes: {self.d_group_berden}", func_n=func_n)
        return
    
    
    def __call__(self, groupby:Literal['slides','samples']):

        self._get_last_exp()
        self.get_unique_slides_from_folder()
        self.get_crops_classes()
        self._count_sample_classes()
        self._count_slide_classes()
        self.compute_berden_score(groupby=groupby)
        return self.d_group_berden 



if __name__ == '__main__':
    computer = BerdenScoreComputer('/Users/marco/yolo/code/helical/config_tcd.yaml')
    computer(groupby='slides')
