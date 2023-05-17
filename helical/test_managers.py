# test managers
from manager import Manager
from utils import get_config_params


def test_yolo_processor(): 

    PARAMS = get_config_params('yolo_processor')
    src_root = PARAMS['src_root']
    dst_root = PARAMS['dst_root']
    slide_format = PARAMS['slide_format']
    label_format = PARAMS['label_format']
    split_ratio = PARAMS['split_ratio']
    data_source = PARAMS['data_source']
    task = PARAMS['task']
    verbose = PARAMS['verbose']
    safe_copy = PARAMS['safe_copy']
    tiling_shape = PARAMS['tiling_shape']
    tiling_shape = (2048,2048)
    tiling_step = PARAMS['tiling_step']
    tiling_level = PARAMS['tiling_level']
    tiling_show = PARAMS['tiling_show']
    stain = PARAMS['stain']
    multiple_samples = PARAMS['multiple_samples']
    resize = PARAMS['resize']
    map_classes = PARAMS['map_classes']
    resize = (512,512)
    manager = Manager(data_source=data_source, src_root=src_root, dst_root=dst_root, 
                      slide_format=slide_format, label_format=label_format, 
                      split_ratio=split_ratio, tiling_shape=tiling_shape,
                      tiling_step=tiling_step, task=task, tiling_level=tiling_level, 
                      stain=stain, tiling_show=tiling_show, verbose=verbose, 
                      safe_copy=safe_copy, multiple_samples=multiple_samples, 
                      resize = resize, map_classes=map_classes)
    manager()

    return







if __name__ == '__main__':
    test_yolo_processor()
        