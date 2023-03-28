from plotter_data_detect_muw_sfog import PlotterDetectMUW
from plotter_data_detect_hubmap_pas import PlotterDetectHub


def test_plotter_data_detect_muw_sfog():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_plotter_detect_muw_sfog/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    # files = ['/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_31_18.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_52_12.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_49_24.png']
    plotter = PlotterDetectMUW(data_root=data_root, files=None, verbose = False)
    plotter()

    return

def test_plotter_data_detect_hubmap_pas():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_plotter_detect_hubmap_pas/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    wsi_images_like = '*.tif'
    verbose = True
    files = None
    wsi_labels_like = '*.json'
    tile_images_like = '*.png'
    tile_labels_like = '*.txt'
    empty_ok=True
    # files = ['/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_31_18.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_52_12.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_49_24.png']
    plotter = PlotterDetectHub(data_root=data_root, files=files, verbose = verbose, wsi_images_like = wsi_images_like, 
                               wsi_labels_like = wsi_labels_like, tile_images_like = tile_images_like,
                               tile_labels_like = tile_labels_like, empty_ok=empty_ok) 
    plotter()

    return

if __name__ == '__main__':
    test_plotter_data_detect_muw_sfog()