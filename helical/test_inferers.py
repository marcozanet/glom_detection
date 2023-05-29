from yolo_detector_muw_sfog_infere2 import YOLO_Inferer_Detector
from utils import get_config_params
import os

def test_yolo_infere_detect_muw_sfog(): 

    PARAMS = get_config_params('test_yolo_infere_detect_muw_sfog')
    repository_dir = PARAMS['repository_dir']
    yolov5dir = PARAMS['yolov5dir']
    images_dir = PARAMS['images_dir']
    weights = PARAMS['weights']
    save_crop = PARAMS['save_crop']
    augment = PARAMS['augment']
    conf_thres = PARAMS['conf_thres']
    repository_dir = PARAMS['repository_dir']
    visualize = PARAMS['visualize']
    save_txt = PARAMS['save_txt']
    detector = YOLO_Inferer_Detector(images_dir = images_dir, weights = weights, yolov5dir=yolov5dir,
                                    repository_dir=repository_dir, augment=augment, conf_thres = conf_thres, 
                                    save_crop=save_crop)
    detector.infere(visualize=visualize, save_txt=save_txt)

    return

def test_yolo_infere_trainvaltest_muw_sfog_for_feature_extraction(): 
    """ Does yolo inference on train, val and test set (output in 3 detect exp folds) 
        so as to then create a dataset for the cnn training. """
    
    PARAMS = get_config_params('test_yolo_infere_detect_muw_sfog')
    repository_dir = PARAMS['repository_dir']
    yolov5dir = PARAMS['yolov5dir']
    images_dir = PARAMS['images_dir'] 
    datasets = ['test', 'val', 'train']
    weights = PARAMS['weights'] 
    save_crop = PARAMS['save_crop']
    augment = PARAMS['augment']
    conf_thres = PARAMS['conf_thres']
    repository_dir = PARAMS['repository_dir']
    visualize = PARAMS['visualize']
    save_txt = PARAMS['save_txt']

    assert os.path.split(os.path.dirname(images_dir))[1] in datasets, f"dirname is {os.path.dirname(images_dir)} but should be train, val or test."
    change_set = lambda fp, _set: os.path.join(os.path.dirname(os.path.dirname(fp)), _set, 'images')
    
    for _set in datasets: 
        print(f"Infering on {_set}.")
        print("-"*20)
        images_dir = change_set(images_dir, _set)
        detector = YOLO_Inferer_Detector(images_dir = images_dir, weights = weights, yolov5dir=yolov5dir,
                                        repository_dir=repository_dir, augment=augment, conf_thres = conf_thres, 
                                        save_crop=save_crop)
        detector.infere(visualize=visualize, save_txt=save_txt)

    return

def test_yolo_infere_detect_hubmap_pas(): 

    PARAMS = get_config_params('test_yolo_infere_detect_hubmap_pas')
    repository_dir = PARAMS['repository_dir']
    yolov5dir = PARAMS['yolov5dir']
    images_dir = PARAMS['images_dir']
    weights = PARAMS['weights']
    save_crop = PARAMS['save_crop']
    augment = PARAMS['augment']
    conf_thres = PARAMS['conf_thres']
    repository_dir = PARAMS['repository_dir']
    visualize = PARAMS['visualize']
    save_txt = PARAMS['save_txt']

    detector = YOLO_Inferer_Detector(images_dir = images_dir, weights = weights, yolov5dir=yolov5dir,
                                    repository_dir=repository_dir, augment=augment, conf_thres = conf_thres)
    detector.infere(visualize=visualize, save_txt=save_txt)

    return

if __name__ == "__main__": 
    test_yolo_infere_trainvaltest_muw_sfog_for_feature_extraction()