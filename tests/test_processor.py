import sys
sys.path.append("/Users/marco/yolo/code")

from helical.processor_tile import TileProcessor

params = {'src_root':'/Users/marco/glomseg-share', 
          'dst_root': '/Users/marco/datasets/muw_exps', 
          'mode': 'detection', 
          'ratio':[0.7, 0.15, 0.15]}


def test_get_trainvaltest(): 

    processor = Processor(src_root=params['src_root'],
                          dst_root=params['dst_root'],
                          mode = params['mode'], 
                          ratio=params['ratio'])
    processor.get_trainvaltest()

    return

def test_make_bboxes():

    processor = Processor(src_root=params['src_root'],
                          dst_root=params['dst_root'],
                          mode = params['mode'], 
                          ratio=params['ratio'])
    processor.make_bboxes(clear_txt=True,
                          reduce_classes=True)


    return

def test_processor():
    processor = Processor(src_root=params['src_root'],
                          dst_root=params['dst_root'],
                          mode = params['mode'], 
                          ratio=params['ratio'])
    processor.get_trainvaltest()
    processor.make_bboxes()

    return




if __name__ == '__main__':
    # test_get_trainvaltest()
    # test_make_bboxes()
    print('nothing to see here.')