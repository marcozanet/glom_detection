from yolo_detector_train_muw_sfog import YOLODetector

def test_yolo_detector_train_muw_sfog(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_yolo_detector_train_muw_sfog/detection/tiles' if system == 'mac' else r'D:\marco\datasets\slides\detection\tiles'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} 
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=3 
    dataset = 'muw'
    detector = YOLODetector(dataset = dataset,
                            data_folder=data_folder, 
                            repository_dir=repository_dir,
                            yolov5dir=yolov5dir,
                            map_classes=map_classes,
                            tile_size = tile_size,
                            batch_size=batch_size,
                            epochs=epochs,
                            workers=workers,
                            device=device,
                            save_features=save_features)
    detector.train()

    return
        



if __name__ == '__main__':
    
    test_yolo_detector_train_muw_sfog()
