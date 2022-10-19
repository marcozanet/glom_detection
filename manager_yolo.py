import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
 
from data_preparation_yolo import *
from data_preparation_unet import *
from unet_utils import split_sets
from yolo_utils import *
from tqdm import tqdm
from typing import List, Type
from unet_runner import GlomModel, create_dirs, get_loaders
import pytorch_lightning as pl
from crop_predictions import crop_obj
import predict_unet as pu
from reconstruct_tile import reconstruct
import utils
import shutil
import torch

# TODO try creating environment with pytorch for m1, requirements for yolo, 
# requirements for segmentation model pytorch (including pytorchlightning)

class Manager():
    def __init__(self, 
                src_folder:str,
                dst_folder: str,
                mode: str,
                system: str = 'windows',
                ratio: float = [0.7, 0.15, 0.15], 
                tile_shape = 2048,
                yolo_tiles = 512,
                yolo_batch = 8,
                yolo_epochs = 20,
                unet_tiles = 512,
                unet_epochs = 10,
                yolo_weights = False,
                unet_ratio: List[float] = [0.7, 0.15, 0.15],
                ) -> None: 

        # check if train, val, test already exist; if so, then 
        # if mode == 'train':
        #     if not os.path.isdir(root):
        #         os.makedirs(root)
        #     dirs = 'train', 'val', 'test'
        #     already_splitted = True
        #     for dir in dirs:
        #         if not os.path.isdir(os.path.join(root, dir)):
        #             already_splitted = False
        #     if already_splitted is False:
        #         self.train_dir, self.val_dir, self.test_dir = split_wsi_yolo(data_folder= src_folder, new_root=dst_folder, ratio = ratio)
        #     else:
        #         self.train_dir = [os.path.join(dst_folder, dir) for dir in os.listdir(root) if 'train' in dir][0]
        #         self.val_dir = [os.path.join(dst_folder, dir) for dir in os.listdir(root) if 'val' in dir][0]
        #         self.test_dir = [os.path.join(dst_folder, dir) for dir in os.listdir(root) if 'test' in dir][0]
        # elif mode == 'test':
        #     if not os.path.isdir(dst_folder):
        #         os.makedirs(dst_folder)


        self.src_dir = src_folder
        self.dst_dir = dst_folder
        self.root = dst_folder
        self.mode = mode
        self.tile_shape = tile_shape
        self.yolo_tiles = yolo_tiles
        self.yolo_batch = yolo_batch
        self.yolo_epochs = yolo_epochs
        self.unet_tiles = unet_tiles
        self.yolo_weights = yolo_weights
        self.ratio = ratio
        self.unet_epochs = unet_epochs
        self.system = system

        if self.system == 'windows':
            self.yolov5dir = 'C:\marco\yolov5'
            self.yolodir = 'C:\marco\code\glom_detection'
        elif self.system == 'mac':
            self.yolodir = '/Users/marco/yolo'
            self.yolov5dir = '/Users/marco/yolov5'


        if mode == 'test':
            self.slides_fns = self.set_folders()
        elif mode == 'train':
            masks_already_computed = utils.check_already_patchified(train_dir= os.path.join(self.dst_dir, 'training', 'train') )
            print(masks_already_computed)
            if masks_already_computed is False:
                self.train_dir, self.val_dir, self.test_dir =  self.set_folders()
            else:
                self.train_dir = os.path.join(self.dst_dir, 'training', 'train')
                self.val_dir = os.path.join(self.dst_dir, 'training', 'val')
                self.test_dir = os.path.join(self.dst_dir, 'training', 'test')
            # TODO creare una funzione riprendi_da in modo da non rifare sempre tutto dall'inizio cancellando ecc
        
        self.masks_already_computed = masks_already_computed

        return
    

    def set_folders(self):

        if self.mode == 'train':
            target_dir = os.path.join(self.dst_dir, 'training')
                # if exists, collect all tiff and move them back to source before deleting
            if os.path.isdir(target_dir):
                wsi_files_tot = []
                for (root, _, files) in os.walk(os.path.join(self.dst_dir, 'training')):
                    wsi_files = [file for file in files if '.tiff' in file or '.json' in file and '.DS_Store' not in file]
                    wsi_files = [os.path.join(root, file) for file in wsi_files]
                    wsi_files_tot.extend(wsi_files)
                for fp in wsi_files_tot:
                    fn = os.path.split(fp)[1]
                    os.rename(src = os.path.join(fp), dst = os.path.join(self.src_dir, fn))
            shutil.rmtree(target_dir)

            train_wsis, val_wsis, test_wsis = utils.split_slides(data_folder = self.src_dir, ratio = self.ratio)
            print(f"train: {train_wsis}, \nval = {val_wsis}, \ntest: {test_wsis}")
            dirs = [train_wsis, val_wsis, test_wsis]
            dir_names = ['train', 'val', 'test']
            for slides, dir_name in zip(dirs, dir_names):
                subdirs =  ['bb', 'images', 'masks']
                for fn in slides:
                    tiles_dir = [os.path.join(self.dst_dir, 'training', dir_name, fn, 'tiles', subdir) for subdir in subdirs ]
                    masks_dir = [os.path.join(self.dst_dir, 'training', dir_name, fn, 'crops', subdir) for subdir in subdirs ]
                    for tile_dir, mask_dir in zip(tiles_dir, masks_dir):
                        if not os.path.isdir(tile_dir):
                            os.makedirs(tile_dir)
                        if not os.path.isdir(mask_dir):
                            os.makedirs(mask_dir)  
            
            train_dir = os.path.join(self.dst_dir, 'training', 'train')
            val_dir = os.path.join(self.dst_dir, 'training', 'val')
            test_dir = os.path.join(self.dst_dir, 'training', 'test')

            utils.move_wsis(src_dir = self.src_dir, root_dir = self.dst_dir, mode = 'forth')

            return train_dir, val_dir, test_dir
        
        elif self.mode == 'test':
            wsi_fns = list(set([file.split('.')[0] for file in os.listdir(self.src_dir) if 'tiff' in file and 'DS' not in file]))
            subdirs =  ['bb', 'images', 'masks']
            for fn in wsi_fns:
                tiles_dir = [os.path.join(self.dst_dir, 'predictions', fn, 'tiles', subdir) for subdir in subdirs ]
                masks_dir = [os.path.join(self.dst_dir, 'predictions', fn, 'crops', subdir) for subdir in subdirs ]
                for tile_dir, mask_dir in zip(tiles_dir, masks_dir):
                    if not os.path.isdir(tile_dir):
                        os.makedirs(tile_dir)
                    if not os.path.isdir(mask_dir):
                        os.makedirs(mask_dir)
            return 
        else:
            raise TypeError(f"'mode' should be either 'train' or 'test'. ")


        return 

    def prepare_training_yolo(self):

        if self.masks_already_computed is True:
            print("Preparation to train YOLO already done. ")
            return

        roots = [self.train_dir, self.val_dir, self.test_dir]
        for root in roots:
            dirs = [os.path.join(root, dir) for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]
            for dir in dirs:
                print(f"Creating bb annotations for {dir}:")
                get_wsi_bb(source_folder=dir, convert_to= 'yolo')
                print(f"Creating patches:")
                get_tiles_bb(folder = os.path.join(dir), shape_patch = self.tile_shape)
                print(f"Moving tiles:")
                move_tiles(src_folder= dir, mode = 'train')
                print('Creating tiles:')
                print(f'outfolder: {root}')
                get_tiles_masks(slide_folder = dir, out_folder = root, tile_shape = 2048, tile_step = 2048)
        # print(f"Moving created tile boundig boxes to their folders: ")
        # move_tiles(dir)
        
        return
    

    def train_yolo(self):
        """   Runs the YOLO model as if it was executed on the command line.  """
        
        # move data
        utils.move_yolo(train_dir=self.train_dir, val_dir = self.val_dir, test_dir = self.test_dir, mode = 'forth')
        utils.edit_yaml(root = os.path.join(self.dst_dir, 'training'), mode = 'train')
        os.chdir(self.yolov5dir)
        os.system(f' python train.py --img {self.yolo_tiles} --batch {self.yolo_batch} --epochs {self.yolo_epochs} --data hubmap.yaml --weights yolov5s.pt')
        os.chdir(self.yolodir)
        utils.move_yolo(train_dir=self.train_dir, val_dir = self.val_dir, test_dir = self.test_dir, mode = 'back')

        return
    
    def test_yolo(self):
        """ Tests YOLO on the test set and returns performance metrics. """

        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = get_last_weights()
        else:
            weights_dir = self.yolo_weights
        os.system(f'python val.py --task test --weights {weights_dir} --data data/hubmap.yaml --device cpu')
        os.chdir(self.yolodir)

        return

    

    def predict_yolo(self, dir):
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = get_last_weights()
        else:
            weights_dir = self.yolo_weights

        print(f'Loading weights from {weights_dir}')
        os.system(f'python detect.py --source {dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --conf-thres 0.5 --save-txt --class 0')
        os.chdir(self.yolodir)

        return
    

    def prepare_unet(self, tiles_imgs_folder:str = False):

        # 1- detection with the YOLO on test folder
        # self.predict_yolo(dir = self.train_dir.replace('wsis', 'tiles'))
        # 2 - get results 

        if self.mode == 'test':
            if tiles_imgs_folder is False:
                raise TypeError("''tiles_imgs_folder' can't be False if mode is 'test'")
            txt_folder = get_last_detect()
            save_imgs_folder = tiles_imgs_folder.replace('tiles', 'crops')
            print(f'Cropping images from: {txt_folder} and {tiles_imgs_folder}')
            print(f'Saving images in {save_imgs_folder}')
            crop_obj(txt_folder= txt_folder, 
                     tiles_imgs_folder= tiles_imgs_folder, 
                     save_imgs_folder = save_imgs_folder,
                     crop_shape = 512)

        # 4 - create images and masks 
        if self.mode == 'train':
            txt_folder = get_last_detect()
            utils.move_yolo(train_dir=self.train_dir, 
                            val_dir = self.val_dir, 
                            test_dir = self.test_dir, 
                            mode = 'forth')
            utils.move_wsis(src_dir = self.src_dir,
                            mode = 'back', 
                            root_dir = self.dst_dir )

            root_name = os.path.split(os.path.split(os.path.split(tiles_imgs_folder)[0])[0])[1]
            # print(root_name)
            # print(tiles_imgs_folder)
            save_imgs_folder = os.path.join(os.path.split(tiles_imgs_folder)[0], 'crops', 'images')
            # print(save_imgs_folder)
            print(f'Cropping images from: {txt_folder} and {tiles_imgs_folder}')
            print(f'Saving images in {save_imgs_folder}')
            crop_obj(txt_folder= txt_folder, 
                    tiles_imgs_folder= tiles_imgs_folder, 
                    save_imgs_folder = save_imgs_folder,
                    crop_shape = 512)




            # for dir in [self.train_dir, self.val_dir]:
            #     out_folder = '/Users/marco/hubmap/unet_data'
            #     get_tiles_masks(dir, out_folder)


        # 5 - crop images and masks using the detected images 

        
        # 6 - split cropped images and masks into train, val, test sets
        # if self.mode == 'train':
        #     split_sets(img_dir = tiles_imgs_folder)

        
        return
    
    
    def predict_unet(self, 
                    image_folder: str,
                    coords_folder: str, 
                    crops_folder: str,
                    reconstructed_tiles_folder: str
                    ):
        """ Segments images using pretrained U-Net on test folder. """

        preds_folder = pu.predict(image_folder, plot = False, save_plot_every= 2)
        reconstruct(preds_folder = preds_folder,
                    coords_folder = coords_folder ,
                    crops_folder = preds_folder,
                    reconstructed_tiles_folder= reconstructed_tiles_folder,
                    plot = True )


        return
    
    def train_unet(self):
        """  Trains the U-Net model.  """

        # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(device)
        model = GlomModel(
            arch = 'unet',
            encoder_name='resnet34', 
            encoder_weights='imagenet',
            in_channels = 3,
            out_classes = 1)
            # aux_params = aux_params)
        # create_dirs()
        train_img_dir = os.path.join(self.train_dir, 'model_train', 'images')
        val_img_dir = os.path.join(self.val_dir, 'model_val', 'crops', 'images')
        test_img_dir = os.path.join(self.test_dir, 'model_test', 'crops', 'images')

        train_loader, val_loader, _ = get_loaders(train_img_dir, val_img_dir, test_img_dir)
        trainer = pl.Trainer( max_epochs = self.unet_epochs, )
        trainer.fit(model,
                    train_dataloaders = train_loader, 
                    val_dataloaders = val_loader)

        # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        
        return
    

    def train(self):
        """ Runs the whole pipeline end2end, i.e. runs both YOLO and U-Net. """

        # if self.masks_already_computed is False:
        self.prepare_training_yolo()
        self.train_yolo() # TODO ADD CHECK IF ALREADY TRAINED YOLO
        # dirs = [self.train_dir, self.val_dir, self.test_dir]
        # dirnames = ['train', 'val', 'test']
        # for dir, dirname in zip(dirs, dirnames):
        #     tiles_imgs_folder = os.path.join(dir, f"model_{dirname}", 'images')
        #     print(f"Using YOLO to detect {dirname} images")
        #     self.predict_yolo(dir = tiles_imgs_folder  )
        #     print(f"Preparing {dirname} data for U-Net training:  ")
        #     self.prepare_unet(tiles_imgs_folder= tiles_imgs_folder)
        # self.train_unet()

        return


    def prepare_predicting_yolo(self):
        """ Prepares data for YOLO. """

        print(f"Creating bb annotations for wsis in '{self.src_dir}' folder:")
        get_wsi_bb(source_folder=self.src_dir , convert_to= 'yolo')
        print(f"Creating bb annotations for tiles in {self.src_dir} folder:")
        get_tiles_bb(folder = self.src_dir, shape_patch = self.tile_shape)
        print(f"Moving created tiles and annotations to their folders: ")
        move_tiles(src_folder = self.src_dir, dst_folder =self.dst_dir, mode = 'test')
        print(f'generating slides for: {self.slides_fns}')
        dst = os.path.join(self.dst_dir, 'predictions')
        get_tiles_masks(self.src_dir, out_folder = dst)
        edit_yaml(root = self.dst_dir)

        return


    def predict(self):
        """   Predicts using the entire pipeline.   """

        # self.prepare_predicting_yolo()
        for slide in self.slides_fns:
            # print("Predicting with YOLO on {slide}:")
            # self.predict_yolo( dir = os.path.join(self.dst_dir, 'predictions', slide, 'tiles', 'images'))
            print(f"Preparing U-Net for {slide}:")
            self.prepare_unet(tiles_imgs_folder = os.path.join(self.dst_dir, 'predictions', f'{slide}', 'tiles', 'images'))
            print(f"Segmenting {slide} with U-Net.")
            self.predict_unet(image_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'images'),
                             coords_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'bb') ,
                             crops_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'images'),
                             reconstructed_tiles_folder= os.path.join(self.dst_dir, 'predictions', slide, 'tiles', 'reconstruced'))

        return
        




if __name__ == '__main__':
    
    def get_folders(system = 'windows'):
        if system == 'windows':
            src_folder = r'C:\marco\biopsies\hubmap\slides'
            dst_folder = r'C:\marco\biopsies\hubmap'
        elif system == 'mac':
            src_folder = '/Users/marco/Downloads/train-3'        
            dst_folder = '/Users/marco/hubmap'
        
        return src_folder, dst_folder

    src_folder, dst_folder = get_folders('windows')
    manager = Manager(src_folder = src_folder,
                      dst_folder = dst_folder,
                      ratio = [0.7, 0.15, 0.15], 
                      mode = 'train')
    manager.train()
    # print(manager.train_dir)
    # manager.predict_yolo(dir = '/Users/marco/hubmap/training/train/model_train/images')
    # manager.prepare_unet(tiles_imgs_folder= '/Users/marco/hubmap/training/train/model_train/images' )
    # manager.train()
    # TODO NB SE USI PIOU' VOLTE PREPARE UNET STAI APPENDENDO NUOVI VALORI AI FILE TXT E QUINDI VA TUTTO A TROIE