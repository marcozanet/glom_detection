import sys
sys.path.append("/Users/marco/yolo/code")
import unittest 
from helical.processor_tile import TileProcessor

params = {'src_root':'/Users/marco/glomseg-share', 
          'dst_root': '/Users/marco/Downloads/folder_random', 
          'task': 'segmentation', 
          'ratio':[0.7, 0.15, 0.15]}


class TestTileProcessor(unittest.TestCase):

    def test_get_images_masks(self):

        processor = TileProcessor(src_root=params['src_root'],
                                  dst_root=params['dst_root'],
                                  task = params['task'],
                                  ratio=params['ratio'],
                                  copy = True)
        
        image_list, mask_list = processor._get_images_masks()

    def test_split_images_trainvaltest(self):
        
        processor = TileProcessor(src_root=params['src_root'],
                                  dst_root=params['dst_root'],
                                  task = params['task'],
                                  ratio=params['ratio'], 
                                  copy = True)
        
        image_list, mask_list = processor._get_images_masks()
        images, masks = processor._split_images_trainvaltest(image_list = image_list, mask_list= mask_list)

    def test_get_trainvaltest(self):
    
        processor = TileProcessor(src_root=params['src_root'],
                                  dst_root=params['dst_root'],
                                  task = params['task'],
                                  ratio=params['ratio'],
                                  copy = False)

        processor.get_trainvaltest()  



if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)