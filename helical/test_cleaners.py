import os 
from cleaner_hubmap_segm import CleanerSegmHubmap
from cleaner_muw import CleanerMuw


def test_cleaner_segm_hubmap_pas():
    safe_copy = False
    data_root = '/Users/marco/helical_tests/test_cleaner_segm_hubmap_pas/detection'
    cleaner = CleanerSegmHubmap(data_root=data_root, 
                                safe_copy=safe_copy,
                                wsi_images_like = '*.tif', 
                                wsi_labels_like = '*.json',
                                tile_images_like = '*.png',
                                tile_labels_like = '*.txt',)
    cleaner()

    return

def test_cleaner_detect_muw_sfog():
    safe_copy = False
    data_root = '/Users/marco/helical_datasets/muw_sfog/detection'
    cleaner = CleanerMuw(data_root=data_root, 
                        safe_copy=safe_copy,
                        wsi_images_like = '*.tif', 
                        wsi_labels_like = '*.gson',
                        tile_images_like = '*.png',
                        tile_labels_like = '*.txt',)
    cleaner()

    return


if __name__ == '__main__': 

    test_cleaner_detect_muw_sfog()