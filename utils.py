import os
OPENSLIDE_PATH = r'C:\Users\hp\Documents\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import yaml



def split_slides(data_folder: str, ratio = [0.7, 0.15, 0.15]):
    """ Splits slides and gives back train list fnames, val list fnames, test list fnames. """
    # TODO SAREBBE ANCHE DA VERIFICARE CHE OLTRE A ESSERE LEGGIBILI SIANO ANCHE IN RGB FORMAT E NON GRAYSCALE

    wsi_fns = list(set([file for file in os.listdir(data_folder) if '.tiff' in file and 'DS' not in file]))
    wsi_fps = [os.path.join(data_folder, file) for file in wsi_fns]
    readable_slides = []
    for wsi in wsi_fps:
        try:
            openslide.OpenSlide(wsi)
            readable_slides.append(wsi)
        except:
            print(f"Warning: couldn't read slide {wsi}")

    wsi_fps = readable_slides
    wsi_fns = [os.path.split(file)[1].split('.')[0] for file in wsi_fps]
    # print(f"Slides found: {wsi_fns}. ")

    # split the WSI names between train, val, test
    n_slides = len(wsi_fns)
    train_idx, val_idx = int(ratio[0] * n_slides), max(int((ratio[0] + ratio[1]) * n_slides), 1)
    train_wsis, val_wsis, test_wsis = wsi_fns[:train_idx], wsi_fns[train_idx:val_idx], wsi_fns[val_idx:]
    train_masks = [file.replace('tiff', 'json') for file in train_wsis]
    val_masks = [file.replace('tiff', 'json') for file in val_wsis]
    test_masks = [file.replace('tiff', 'json') for file in test_wsis]
    # slides = [train_wsis, val_wsis, test_wsis]
    # dirs = [wsis_folder_train, wsis_folder_val, wsis_folder_test]
    # masks = [train_masks, val_masks, test_masks]

    return train_wsis, val_wsis, test_wsis


def move_wsis(src_dir: str, root_dir: str, move_test = True, mode = 'forth'):
    """ Moves WSIs and masks to their folders. """

    if mode == 'forth':
        print('Moving WSIs forth')
        train_dir = os.path.join(root_dir, 'training', 'train')
        val_dir = os.path.join(root_dir, 'training', 'val')
        test_dir = os.path.join(root_dir, 'training',  'test')
        dirs = [train_dir, val_dir, test_dir] if move_test is True else [train_dir, val_dir]
        for dir in dirs:
            wsi_dirs = [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir, file)) and 'model' not in file and 'masks' not in file and 'images' not in file]
            wsi_fns = [file + '.tiff' for file in wsi_dirs]
            srcs_imgs = [os.path.join(src_dir, file) for file in wsi_fns]
            dsts_imgs = [os.path.join(dir, wsi_dir, fn) for wsi_dir, fn in zip(wsi_dirs, wsi_fns)]
            
            for src_img, dst_img in zip(srcs_imgs, dsts_imgs):
                os.rename(src_img, dst_img)
            srcs_masks = [file.replace('tiff', 'json') for file in srcs_imgs]
            dsts_masks = [file.replace('tiff', 'json') for file in dsts_imgs]
            for src_mask, dst_mask in zip(srcs_masks, dsts_masks):
                os.rename(src_mask, dst_mask)
    if mode == 'back':
        print('Moving WSIs back')
        src_fps = []
        src_fns = []
        for root, dirs, files in os.walk(os.path.join(root_dir, 'training')):
            wsi_fns = [file for file in files if '.tiff' in file or ".json" in file and "DS" not in file]
            wsi_fps = [os.path.join(root, file) for file in wsi_fns]
            src_fns.extend(wsi_fns)
            src_fps.extend(wsi_fps)
        dst_fps = [os.path.join(src_dir, file) for file in src_fns]

        for src, dst in zip(src_fps, dst_fps):
            os.rename(src = src, dst = dst)
        
    
    return

# def move_yolo_data_temp(train_dir, val_dir, test_dir, mode = 'back'):
#     """ YOLO needs data to be in train, val, test folders:"""

#     dirs = [train_dir, val_dir, test_dir]
#     dir_names = ['train', 'val', 'test']
#     for dir, dirname in zip(dirs, dir_names):
#         yolo_img_dir = os.path.join(dir, f'model_{dirname}', 'images')
#         yolo_bb_dir = os.path.join(dir, f'model_{dirname}', 'labels')
#         os.makedirs(yolo_img_dir, exist_ok= True)
#         os.makedirs(yolo_bb_dir, exist_ok= True)
#         subdirs = [fold for fold in os.listdir(dir) if os.path.isdir(os.path.join(dir, fold)) and 'DS' not in fold and 'model' not in fold] # i.e. if dir not empty
#         subdirs = [fold for fold in subdirs if len(os.listdir(os.path.join(dir, fold))) >= 0 and 'images' not in fold and 'masks' not in fold and 'yolo' not in fold]
#         # print(subdirs)
#         for subdir in subdirs:
#             src_bbs = os.path.join(dir, subdir, 'tiles', 'bb')
#             src_imgs = os.path.join(dir, subdir, 'tiles', 'images')
#             imgs_files = [file for file in os.listdir(src_imgs) if 'png' in file and "DS" not in file]
#             bb_files = [file for file in os.listdir(src_bbs) if 'txt' in file and "DS" not in file]

#             # print(imgs_files)
#             print(bb_files)
#             for img, bb in zip(imgs_files, bb_files):
#                 print(img, bb)
#                 src_img = os.path.join(src_imgs, img)
#                 src_bb = os.path.join(src_bbs, bb)
#                 print(src_img)
#                 print(src_bb)
#                 dst_img = os.path.join(yolo_img_dir, img)
#                 dst_bb = os.path.join(yolo_bb_dir, bb)
#                 print(f"src: {src_img}, dst: {dst_img}")
#                 print(f"src: {src_bb}, dst: {dst_bb}")

#                 # os.rename(src = src_img, dst = dst_img)
#                 # os.rename(src = src_bb, dst = dst_bb)

#     return


def move_yolo(train_dir, val_dir, test_dir, mode = 'forth'):
    """ Opens images, labels from model_train, model_val, model_test and moves them back  """

    dirs = [train_dir, val_dir, test_dir]
    dir_names = ['train', 'val', 'test']
    for dir, dirname in zip(dirs, dir_names):
        model_img_dir = os.path.join(dir, f'model_{dirname}', 'images')
        model_bb_dir = os.path.join(dir, f'model_{dirname}', 'labels')
        model_mask_dir = os.path.join(dir, f'model_{dirname}', 'masks')

        os.makedirs(model_img_dir, exist_ok= True)
        os.makedirs(model_bb_dir, exist_ok= True)
        os.makedirs(model_mask_dir, exist_ok= True)

        subdirs = [fold for fold in os.listdir(dir) if os.path.isdir(os.path.join(dir, fold)) and 'DS' not in fold and 'model' not in fold] # i.e. if dir not empty
        subdirs = [fold for fold in subdirs if len(os.listdir(os.path.join(dir, fold))) >= 0 and 'images' not in fold and 'masks' not in fold and 'model' not in fold]
        # print(subdirs)
        for subdir in subdirs:
            src_bbs = os.path.join(dir, subdir, 'tiles', 'bb')
            src_imgs = os.path.join(dir, subdir, 'tiles', 'images')
            src_masks = os.path.join(dir,  subdir, 'tiles', 'masks')

            
            if mode == 'forth':
                imgs_files = [file for file in os.listdir(src_imgs) if 'png' in file and "DS" not in file]
                bb_files = [file for file in os.listdir(src_bbs) if 'txt' in file and "DS" not in file]
                masks_files = [file for file in os.listdir(src_masks) if 'png' in file and "DS" not in file]

            elif mode == 'back':
                imgs_files = [file for file in os.listdir(model_img_dir) if 'png' in file and "DS" not in file]
                bb_files = [file for file in os.listdir(model_bb_dir) if 'txt' in file and "DS" not in file]
                masks_files = [file for file in os.listdir(model_mask_dir) if 'txt' in file and "DS" not in file]
            
            for img in (imgs_files):
                src_img = os.path.join(src_imgs, img)
                dst_img = os.path.join(model_img_dir, img)
                if mode == 'forth':
                    os.rename(src = src_img, dst = dst_img)
                elif mode == 'back':
                    os.rename(src = dst_img, dst = src_img)
            for mask in (masks_files):
                src_mask = os.path.join(src_masks, mask)
                dst_mask = os.path.join(model_mask_dir, mask)
                if mode == 'forth':
                    os.rename(src = src_mask, dst = dst_mask)
                elif mode == 'back':
                    os.rename(src = dst_mask, dst = src_mask)
            for bb in bb_files:
                src_bb = os.path.join(src_bbs, bb)
                dst_bb = os.path.join(model_bb_dir, bb)
                if mode == 'forth':
                    os.rename(src = src_bb, dst = dst_bb)
                elif mode == 'back':
                    os.rename(src = dst_bb, dst = src_bb)

            # for img, bb in zip(imgs_files, bb_files):
            #     src_img = os.path.join(src_imgs, img)
            #     src_bb = os.path.join(src_bbs, bb)
            #     dst_img = os.path.join(model_img_dir, img)
            #     dst_bb = os.path.join(model_bb_dir, bb)

            #     if mode == 'forth':
            #         os.rename(src = src_img, dst = dst_img)
            #         os.rename(src = src_bb, dst = dst_bb)
            #     elif mode == 'back':
            #         os.rename(src = dst_img, dst = src_img)
            #         os.rename(src = dst_bb, dst = src_bb)
    return


def move_unet(train_dir, val_dir, test_dir, mode = 'forth'):
    """ Opens images, labels from model_train, model_val, model_test and moves them back  """

    dirs = [train_dir, val_dir, test_dir]
    dir_names = ['train', 'val', 'test']
    for dir, dirname in zip(dirs, dir_names):
        model_img_dir = os.path.join(dir, f'model_{dirname}', 'images')
        model_bb_dir = os.path.join(dir, f'model_{dirname}', 'labels')

        unet_images = os.path.join(dir, f'model_{dirname}', 'crops', 'images')
        unet_masks = os.path.join(dir, f'model_{dirname}', 'crops', 'masks')
        
        os.makedirs(model_img_dir, exist_ok= True)
        os.makedirs(model_bb_dir, exist_ok= True)
        os.makedirs(unet_images, exist_ok= True)
        os.makedirs(unet_masks, exist_ok= True)


        subdirs = [fold for fold in os.listdir(dir) if os.path.isdir(os.path.join(dir, fold)) and 'DS' not in fold and 'model' not in fold] # i.e. if dir not empty
        subdirs = [fold for fold in subdirs if len(os.listdir(os.path.join(dir, fold))) >= 0 and 'images' not in fold and 'masks' not in fold and 'model' not in fold]
        for subdir in subdirs:
            src_masks = os.path.join(dir, subdir, 'crops', 'masks')
            src_imgs = os.path.join(dir, subdir, 'crops', 'images')
            print(src_masks)
            if mode == 'forth':
                imgs_files = [file for file in os.listdir(src_imgs) if 'png' in file and "DS" not in file]
                mask_files = [file for file in os.listdir(src_masks) if 'png' in file and "DS" not in file]
                print(imgs_files)
            elif mode == 'back':
                imgs_files = [file for file in os.listdir(model_img_dir) if 'png' in file and "DS" not in file]
                mask_files = [file for file in os.listdir(model_bb_dir) if 'png' in file and "DS" not in file]
            for img, mask in zip(imgs_files, mask_files):
                src_img = os.path.join(src_imgs, img)
                src_mask = os.path.join(src_masks, mask)
                dst_img = os.path.join(unet_images, img)
                dst_mask = os.path.join(unet_masks, mask)

                if mode == 'forth':
                    print(f"src: {dst_img}, dst: {src_img}")
                    print(f"src: {dst_mask}, dst: {src_mask}")
                    # os.rename(src = dst_img, dst = src_img)
                    # os.rename(src = dst_mask, dst = src_mask)
                elif mode == 'back':
                    print(f"src: {src_img}, dst: {dst_img}")
                    print(f"src: {src_mask}, dst: {dst_mask}")                    
                    # os.rename(src = src_img, dst = dst_img)
                    # os.rename(src = src_mask, dst = dst_mask)
    return


def check_already_patchified(train_dir: str):
    
    try:
        slide_dir = os.path.join(train_dir, os.listdir(train_dir))[0]
        if os.path.isdir(slide_dir):
            mask_dir = os.path.join(slide_dir, 'tiles', 'masks')
            masks = os.listdir(mask_dir)
            n_masks = len(masks)
            if n_masks > 2:
                print(f'Found {n_masks} masks in {mask_dir}. Assuming patchification already computed. ')
        computed = True

    except:
        print(f"No masks found while checking destination folder, creating a new tree directory: ")
        computed = False

    return computed

def edit_yaml(root: str = False, test_folder: str = False, mode = 'train' ):
    """ Edits YAML data file from yolov5. """
    if mode == 'test':
        if isinstance(root, str) and test_folder is False:
            yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
            text = {'path':root, 'train': 'train/', 'val':'val/', 'test':'test/'}
            train, val, test = os.path.join(root, 'train'), os.path.join(root, 'val'), os.path.join(root, 'test')
            print(f"YOLO trains and test on: \n-{train} \n-{val} \n-{test}")
            with open(yaml_fp, 'w') as f:
                yaml.dump(data = text, stream=f)
        elif isinstance(test_folder, str) and root is False:
            yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
            print(f"YOLO test on: \n-{test_folder}")

            with open(yaml_fp, 'w') as f:
                yaml.dump(data = text, stream=f)
        else:
            raise TypeError(f"Params to edit_yaml should be path or False and either 'root' or 'test_folder' are to be specified.")
    elif mode == 'train':
        yaml_fp = '/Users/marco/yolov5/data/hubmap.yaml'
        text = {'path':root, 'train': 'train/yolo_train/', 'val':'val/yolo_val/', 'test':'test/yolo_test/', 'names':{0:'glom'}}
        print(text)
        with open(yaml_fp, 'w') as f:
            yaml.dump(data = text, stream=f)

    return

def test_check_already_patchified():
    computed = check_already_patchified('/Users/marco/hubmap/training/train')
    print(computed)
    return
def test_move_yolo_data_temp():
    move_yolo_data_temp(train_dir = '/Users/marco/hubmap/training/train', 
                        val_dir= '/Users/marco/hubmap/training/val',
                        test_dir = '/Users/marco/hubmap/training/test' )
    return
def test_edit_yaml():
    edit_yaml(root = '/Users/marco/hubmap/training' )
    return

def test_move_unet():
    move_unet(train_dir = '/Users/marco/hubmap/training/train', 
              val_dir= '/Users/marco/hubmap/training/val',
              test_dir = '/Users/marco/hubmap/training/test',
              mode = 'forth')
    
    return
def test_move_wsis(mode: str):
    if mode == 'forth':
        move_wsis(root_dir = '/Users/marco/hubmap/',
                  mode = 'forth', 
                  src_dir = '/Users/marco/Downloads/train-3' )
    elif mode == 'back':
        move_wsis(root_dir = '/Users/marco/hubmap/',
                  mode = 'back', 
                  src_dir = '/Users/marco/Downloads/train-3' )

    return

def test_move_yolo():
    move_yolo(train_dir = '/Users/marco/hubmap/training/train', 
              val_dir = '/Users/marco/hubmap/training/val', 
              test_dir = '/Users/marco/hubmap/training/test', 
              mode = 'forth')
    return
if __name__ == '__main__':
    test_move_yolo()
