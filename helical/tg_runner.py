import os, sys 
from tg_inference import YOLO_Inferer
from tg_validate import YOLO_Validator

def run(config_yaml_fp:str, 
        models_yaml_fp:str,
        mode:str):
    """ Main function to run inference or validation based on the config settings in the config yaml file."""

    if mode == 'validation':
        validator = YOLO_Validator(config_yaml_fp=config_yaml_fp,
                                models_yaml_fp=models_yaml_fp)
        validator()
    elif mode == 'inference':
        inferer = YOLO_Inferer(config_yaml_fp=config_yaml_fp,
                               models_yaml_fp=models_yaml_fp)
        inferer()
    else: 
        raise ValueError(f"'mode':{mode} should be either 'validate' or 'infere'")

    return


if __name__ == '__main__':
    config_yaml_fp = '/Users/marco/yolo/code/helical/tg_config.yaml'
    models_yaml_fp= '/Users/marco/yolo/code/helical/trained_models.yaml'
    mode = 'inference'
    run(config_yaml_fp=config_yaml_fp, 
        models_yaml_fp=models_yaml_fp, 
        mode=mode)