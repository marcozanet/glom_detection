import os, shutil
import yaml
import tg_runner
from typing import Any
from utils import get_config_params
import pytest

class TestTGRunner:

    config_yaml_fp = '/Users/marco/yolo/code/helical/tg_config_test.yaml'
    models_yaml_fp= '/Users/marco/yolo/code/helical/trained_models.yaml'


    def _set_config_vals(self, mode:str, key:str, val:Any ):

        with open(self.config_yaml_fp, 'r') as f: 
            params = yaml.load(f, Loader=yaml.SafeLoader)
        
        params[mode][key] = val 
        params = yaml.dump(params)
        with open(self.config_yaml_fp, 'w') as f: 
            f.write(params)

        return
    
    @pytest.mark.parametrize("classify", [False, True])
    @pytest.mark.parametrize("save_txt, exp_dir_out_exist", [(False, False), (True, True) ])
    def test_runner_validation(self, classify:bool, save_txt:bool, exp_dir_out_exist:str):

        MODE='validation'
        params = get_config_params(yaml_fp=self.config_yaml_fp, config_name=MODE)
        self._set_config_vals(mode=MODE, key='classify', val=classify)
        self._set_config_vals(mode=MODE, key='save_txt', val=save_txt)

        if os.path.isdir(params['output_dir']): shutil.rmtree(params['output_dir'])
        tg_runner.run(config_yaml_fp=self.config_yaml_fp,
                      models_yaml_fp=self.models_yaml_fp, 
                      mode=MODE)
        assert os.path.isdir(params['output_dir'])
        if exp_dir_out_exist is True:
            assert os.path.isdir(os.path.join(params['output_dir'], 'labels'))
        else:
            assert not os.path.isdir(os.path.join(params['output_dir'], 'labels'))

        return
    

    @pytest.mark.parametrize("classify", [False, True])
    @pytest.mark.parametrize("save_txt, exp_dirout_labels_exist", [(False, False),(True, True)])
    @pytest.mark.parametrize("save_imgs, exp_outimages_exist", [(False, False), (True, True)])
    def test_runner_inference(self, 
                               classify:bool, 
                               save_txt:bool, exp_dirout_labels_exist:str, 
                               save_imgs:bool, exp_outimages_exist:str ):

        MODE='inference'
        params = get_config_params(yaml_fp=self.config_yaml_fp, config_name=MODE)
        self._set_config_vals(mode=MODE, key='classify', val=classify)
        self._set_config_vals(mode=MODE, key='save_txt', val=save_txt)
        self._set_config_vals(mode=MODE, key='save_imgs', val=save_imgs)
        output_dir = params['output_dir']

        if os.path.isdir(output_dir): shutil.rmtree(params['output_dir'])
        tg_runner.run(config_yaml_fp=self.config_yaml_fp,
                      models_yaml_fp=self.models_yaml_fp, 
                      mode=MODE)
        
        assert os.path.isdir(output_dir)
        if exp_dirout_labels_exist is True:
            assert os.path.isdir(os.path.join(output_dir, 'labels'))
        else:
            assert not os.path.isdir(os.path.join(output_dir, 'labels'))

        made_images = [file for file in os.listdir(output_dir) if '.png' in file]
        if exp_outimages_exist is True:
            assert len(made_images)>0
        else:
            assert len(made_images)==0
        return


