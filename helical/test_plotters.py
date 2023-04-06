from plotter_data_detect_muw_sfog import PlotterDetectMUW
from plotter_data_detect_hubmap_pas import PlotterDetectHub
from plotter_data_segm_hubmap_pas import PlotterSegmHubmap


def test_plotter_data_detect_muw_sfog():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_plotter_detect_muw_sfog/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    plotter = PlotterDetectMUW(data_root=data_root, files=None, verbose = False)
    plotter()

    return

def test_plotter_data_detect_hubmap_pas():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_plotter_detect_hubmap_pas/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    verbose = True
    files = None
    plotter = PlotterDetectHub(data_root=data_root, files=files, verbose = verbose) 
    plotter()

    return


def test_plotter_data_segm_hubmap_pas():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_plotter_segm_hubmap_pas/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    plotter = PlotterSegmHubmap(data_root=data_root, 
                                files=None, 
                                verbose = False) 
    plotter()


    return

# if __name__ == '__main__':
#     test_plotter_data_segm_hubmap_pas()