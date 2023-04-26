from yolo_detect_trainer import YOLO_Trainer_Detector
from yolo_segment_trainer import YOLO_Trainer_Segmentor

def test_yolo_trainer_detect_muw_sfog(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection/tiles' if system == 'mac' else r'D:\marco\datasets\yolo_detect_muw_sfog\detection'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes = {'Glo-healthy':1, 'Glo-unhealthy':0} 
    save_features = False
    tile_size = 128
    batch_size=5 if system == 'mac' else 4
    epochs=1
    dataset = 'muw'
    crossvalid_tot_kfolds = 3
    crossvalid_cur_kfold = 2
    note = 'testing'
    detector = YOLO_Trainer_Detector(dataset = dataset, data_folder=data_folder, repository_dir=repository_dir, 
                                    yolov5dir=yolov5dir, map_classes=map_classes, tile_size = tile_size,
                                    batch_size=batch_size, epochs=epochs, workers=workers, 
                                    device=device, save_features=save_features, crossvalid_tot_kfolds=crossvalid_tot_kfolds,
                                    crossvalid_cur_kfold=crossvalid_cur_kfold, note = note)
    detector.train()

    return
        
def test_yolo_trainer_detect_hub_pas(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_yolo_detect_train_hubmap_pas/detection/tiles' if system == 'mac' else r'D:\marco\datasets\yolo_detect_muw_sfog\detection'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes = {'glomerulus':0} 
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=5
    dataset = 'hubmap'
    crossvalid_tot_kfolds = 3
    crossvalid_cur_kfold = 1
    detector = YOLO_Trainer_Detector(dataset = dataset, data_folder=data_folder, repository_dir=repository_dir, 
                                    yolov5dir=yolov5dir, map_classes=map_classes, tile_size = tile_size,
                                    batch_size=batch_size, epochs=epochs, workers=workers, 
                                    device=device, save_features=save_features,
                                    crossvalid_tot_kfolds=crossvalid_tot_kfolds,
                                    crossvalid_cur_kfold=crossvalid_cur_kfold)
    detector.train()

    return


def test_yolo_trainer_segm_hub_pas(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_yolo_segm_train_hubmap_pas/detection/tiles' if system == 'mac' else r'D:\marco\datasets\yolo_detect_muw_sfog\detection'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes =  {'glomerulus':0} 
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=1
    dataset = 'hubmap'
    crossvalid_tot_kfolds = 3
    crossvalid_cur_kfold = 1
    segmentor = YOLO_Trainer_Segmentor(dataset = dataset, data_folder=data_folder, repository_dir=repository_dir,
                                        yolov5dir=yolov5dir, map_classes=map_classes, tile_size = tile_size,
                                        batch_size=batch_size, epochs=epochs, workers=workers, device=device,
                                        save_features=save_features, crossvalid_tot_kfolds=crossvalid_tot_kfolds,
                                        crossvalid_cur_kfold=crossvalid_cur_kfold)
    segmentor.train()

    return


def test_yolo_trainer_segm_muw_sfog(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    repository_dir = '/Users/marco/yolo/code/helical' if system == 'mac' else r'C:\marco\code\glom_detection\helical'
    yolov5dir = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    data_folder = '/Users/marco/helical_tests/test_yolo_segm_train_hubmap_pas/detection/tiles' if system == 'mac' else 'not implemented'
    device = None if system == 'mac' else 'cuda:0'
    workers = 0 if system == 'mac' else 1
    map_classes =  {'Glo-healthy':1, 'Glo-unhealthy':0} 
    save_features = False
    tile_size = 512 
    batch_size=2 if system == 'mac' else 4
    epochs=1
    dataset = 'hubmap'
    crossvalid_tot_kfolds = 3
    crossvalid_cur_kfold = 1
    segmentor = YOLO_Trainer_Segmentor(dataset = dataset, data_folder=data_folder, repository_dir=repository_dir,
                                        yolov5dir=yolov5dir, map_classes=map_classes, tile_size = tile_size,
                                        batch_size=batch_size, epochs=epochs, workers=workers, device=device,
                                        save_features=save_features, crossvalid_tot_kfolds=crossvalid_tot_kfolds,
                                        crossvalid_cur_kfold=crossvalid_cur_kfold)
    segmentor.train()

    return




if __name__ == '__main__':
    
    test_yolo_trainer_detect_hub_pas()