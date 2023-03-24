
from manager_muw import ManagerMUW
from manager_hubmap import ManagerHubmap


def test_manager_detect_muw_sfog(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    src_root = '/Users/marco/helical_tests/test_manager_muw_sfog/' if system == 'mac' else  r'D:\marco\datasets\slides'
    dst_root = '/Users/marco/helical_tests/test_manager_muw_sfog/' if system == 'mac' else  r'D:\marco\datasets\slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.34, 0.33, 0.33]    
    data_source = 'muw'
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (1024,1024)
    tiling_step = 1024
    tiling_level = 3
    tiling_show = True
    stain = 'sfog'

    manager = ManagerMUW(data_source=data_source, src_root=src_root, dst_root=dst_root, slide_format=slide_format,
                        label_format=label_format, split_ratio=split_ratio, tiling_shape=tiling_shape,
                        tiling_step=tiling_step, task=task, tiling_level=tiling_level, stain=stain,
                        tiling_show=tiling_show, verbose=verbose, safe_copy=safe_copy)
    manager()

    return

def test_manager_detect_muw_pas(): 

    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    src_root = '/Users/marco/helical_tests/test_manager_muw_pas/' if system == 'mac' else  r'D:\marco\datasets\slides'
    dst_root = '/Users/marco/helical_tests/test_manager_muw_pas/' if system == 'mac' else  r'D:\marco\datasets\slides'
    slide_format = 'tif'
    label_format = 'json'
    split_ratio = [0.34, 0.33, 0.33]    
    data_source = 'muw'
    task = 'detection'
    verbose = True
    safe_copy = False
    tiling_shape = (1024,1024)
    tiling_step = 1024
    tiling_level = 3
    tiling_show = True
    stain = 'pas'

    manager = ManagerMUW(data_source=data_source, src_root=src_root, dst_root=dst_root, slide_format=slide_format,
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

    manager = ManagerHubmap(src_root=src_root, dst_root=dst_root, slide_format=slide_format, label_format=label_format,
                            split_ratio=split_ratio, tiling_shape=tiling_shape, tiling_step=tiling_step,
                            task=task, tiling_show=tiling_show, verbose=verbose, safe_copy=safe_copy, stain=stain)
    manager()

    return

if __name__ == '__main__':
    test_manager_detect_hubmap_pas()
        