import sys
sys.path.append("/Users/marco/yolo/code")
import unittest

from helical.converter import Converter

def test_Converter():
    """ Tests converter class. """
    
    folder = '/Users/marco/Downloads/new_source'
    converter = Converter(folder = folder, 
                          convert_from='json_wsi_mask', 
                          convert_to='txt_wsi_bboxes',
                          save_folder= '/Users/marco/Downloads/folder_random' )
    converter()


    return


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # NON SO PERCHE' DA' SOLO 0 TEST RAN