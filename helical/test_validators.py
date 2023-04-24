from yolo_detector_validate import YOLO_Validator_Detector



def test_yolo_val_muw_sfog(): 

    system = 'mac'
    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else 'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else 'C:\marco\yolov5'
    images_dir = '/Users/marco/helical_tests/test_yolo_detect_infere_muw_sfog/val/images'
    weights = '/Users/marco/helical_tests/test_yolo_detect_infere_muw_sfog/best.pt'
    augment = False
    conf_thres=0.304
    detector = YOLO_Validator_Detector(images_dir = images_dir, weights = weights, yolov5dir=yolov5dir,
                                    repository_dir=repository_dir, augment=augment, conf_thres = conf_thres)
    visualize= True
    save_txt=False
    detector.infere(visualize=visualize, save_txt=save_txt)

    return

def test_yolo_infere_detect_hubmap_pas(): 

    system = 'mac'
    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else 'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else 'C:\marco\yolov5'
    images_dir = '/Users/marco/helical_tests/test_yolo_detect_infere_hubmap_pas/test/images'
    weights = '/Users/marco/helical_tests/test_yolo_detect_infere_muw_sfog/best.pt'
    conf_thres=0.304
    augment = False

    detector = YOLO_Validator_Detector(images_dir = images_dir, weights = weights, yolov5dir=yolov5dir,
                                    repository_dir=repository_dir, augment=augment, conf_thres = conf_thres)
    visualize= True
    save_txt=False
    detector.infere(visualize=visualize, save_txt=save_txt)

    return


if __name__ == '__main__':
    
    test_yolo_val_muw_sfog()
