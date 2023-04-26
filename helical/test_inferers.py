from yolo_detector_muw_sfog_infere2 import YOLO_Inferer_Detector
from utils import get_config_params

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
    test_yolo_infere_detect_muw_sfog()