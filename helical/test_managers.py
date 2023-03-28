
from manager_detect_muw_sfog import ManagerDetectMuwSFOG
from manager_detect_hubmap_pas import ManagerDetectHubmapPAS
from manager_segm_hubmap_pas import ManagerSegmHubPAS


def test_manager_detect_muw_sfog(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    src_root = '/Users/marco/helical_tests/test_manager_detect_muw_sfog' if system == 'mac' else  r'D:\marco\datasets\slides'
    dst_root = '/Users/marco/helical_tests/test_manager_detect_muw_sfog' if system == 'mac' else  r'D:\marco\datasets\slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.34, 0.33, 0.33]    
    data_source = 'muw'
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (2048,2048)
    tiling_step = 512
    tiling_level = 3
    tiling_show = True
    stain = 'sfog'

    manager = ManagerDetectMuwSFOG(data_source=data_source, src_root=src_root, dst_root=dst_root, slide_format=slide_format,
                        label_format=label_format, split_ratio=split_ratio, tiling_shape=tiling_shape,
                        tiling_step=tiling_step, task=task, tiling_level=tiling_level, stain=stain,
                        tiling_show=tiling_show, verbose=verbose, safe_copy=safe_copy)
    manager()

    return



def test_manager_detect_hubmap_pas(): 

    # TODO FARE INHERITANCE MANAGER HUBMAP MANAGER BASE
    print('TODO FARE INHERITANCE MANAGER HUBMAP MANAGER BASE')
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    # DEVELOPMENT 
    src_root = '/Users/marco/helical_tests/test_manager_hubmap_pas' if system == 'mac' else  r'D:\marco\hubmap_slides'
    dst_root = '/Users/marco/helical_tests/test_manager_hubmap_pas' if system == 'mac' else  r'D:\marco\hubmap_slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.34, 0.33, 0.33]  # TODO CAMBIAAAAAAA
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (2048,2048)
    tiling_step = 2048  # TODO CAMBIAAAAAAA
    tiling_show = False
    stain = 'pas'

    manager = ManagerDetectHubmapPAS(src_root=src_root, dst_root=dst_root, slide_format=slide_format, label_format=label_format,
                            split_ratio=split_ratio, tiling_shape=tiling_shape, tiling_step=tiling_step,
                            task=task, tiling_show=tiling_show, verbose=verbose, safe_copy=safe_copy, stain=stain)
    manager()

    return

def test_manager_segm_hubmap_pas(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    # DEVELOPMENT 
    src_root = '/Users/marco/helical_tests/test_cleaner_segm_hubmap_pas' if system == 'mac' else  r'D:\marco\hubmap_slides\detection'
    dst_root = '/Users/marco/helical_tests/test_cleaner_segm_hubmap_pas' if system == 'mac' else  r'D:\marco\hubmap_slides\detection'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.34, 0.33, 0.33]    
    data_source = 'hubmap'
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (2048,2048)
    tiling_step = 1024
    tiling_level = 0
    tiling_show = True
    map_classes = {"glomerulus":0}
    inflate_points_ntimes=0


    manager = ManagerSegmHubPAS(data_source=data_source, map_classes=map_classes,src_root=src_root, 
                             dst_root=dst_root, slide_format=slide_format, inflate_points_ntimes=inflate_points_ntimes,
                             label_format=label_format, split_ratio=split_ratio, tiling_shape=tiling_shape,
                             tiling_step=tiling_step, task=task, tiling_level=tiling_level,tiling_show=tiling_show,
                             verbose=verbose, safe_copy=safe_copy)
    manager()

    return


if __name__ == '__main__':
    test_manager_segm_hubmap_pas()
        