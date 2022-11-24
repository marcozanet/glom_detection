import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
 
from data_preparation_yolo import *
from data_preparation_unet import *
import utils_yolo
import utils_unet
import utils_manager
from tqdm import tqdm
from typing import List, Type
from unet_runner import GlomModel, create_dirs, get_loaders
import pytorch_lightning as pl
from crop_predictions import crop_obj
import predict_unet as pu
from reconstruct_tile import reconstruct
from processor import Processor
import shutil
import torch
import time
import datetime
from statistix import plot_labels

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
                yolo_epochs = 3,
                conf_thres = 0.8,
                unet_classes = 3,
                unet_tiles = 512,
                unet_epochs = 20,
                unet_resize = False,
                unet_weights_save_path = '/Users/marco/hubmap/unet/',
                yolo_weights = False,
                yolo_val_augment = False, 
                unet_ratio: List[float] = [0.7, 0.15, 0.15],
                ) -> None: 

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
        self.unet_resize = unet_resize
        self.system = system
        self.unet_classes = unet_classes
        self.yolo_val_augment = yolo_val_augment
        self.conf_thres = 0.8

        self.processor = Processor(src_root=src_folder,
                                   dst_root=dst_folder,
                                   mode='detection',
                                   ratio=ratio)

        if self.system == 'windows':
            self.yolov5dir = 'C:\marco\yolov5'
            self.yolodir = 'C:\marco\code\glom_detection'
            self.code_dir = ''
            raise Exception('define weights path for unet')
        elif self.system == 'mac':
            self.yolodir = '/Users/marco/yolo'
            self.yolov5dir = '/Users/marco/yolov5'
            self.unet_weights_save_path = unet_weights_save_path
            self.code_dir = os.path.join(self.yolodir, 'code')

        if mode == 'test':
            raise NotImplemented() 
            # check if already splitted into train val test 
            self.slides_fns = self.set_folders()
        elif mode == 'train':
            self.train_dir, self.val_dir, self.test_dir =  self.processor.get_trainvaltest()

        return
    

    def train_yolo(self):
        """   Runs the YOLO model as if it was executed on the command line.  """
        
        # 1) show statistics dataset:
        plot_labels(self.train_dir, reduce_classes = True)


        start_time = time.time()
        yolo_classes = {0: 'healthy', 1: 'unhealthy'}
        otherinfo_yolo = {'data': self.src_dir, 'classes': yolo_classes, 'epochs': self.yolo_epochs}
        utils_manager.edit_yaml(root = self.dst_dir, mode = 'train', system = self.system, classes = yolo_classes )
        os.chdir(self.yolov5dir)
        os.system(f' python train.py --img {self.yolo_tiles} --batch {self.yolo_batch} --epochs {self.yolo_epochs} --data hubmap.yaml --weights yolov5s.pt')
        os.chdir(self.yolodir)
        end_time = time.time()

        utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = '/Users/marco/yolov5/runs/train')
        train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        print(f"Training Done. Train YOLO duration: {train_yolo_duration}")

        return  train_yolo_duration

    
    def test_yolo(self, conf_thres = None):
        """ Tests YOLO on the test set and returns performance metrics. """

        assert isinstance(conf_thres, float) or conf_thres is None, TypeError(f"conf_thres is {type(conf_thres)}, but should be either None or float.")

        # get model
        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = self.yolo_weights

        # define command
        command = f'python val.py --task test --weights {weights_dir} --data data/hubmap.yaml --device cpu'
        if self.yolo_val_augment is True:
            command += " --augment"
        if conf_thres is not None:
            command += f" --conf_thres {conf_thres}" 

        # execute
        os.system(f'python val.py --task test --weights {weights_dir} --data data/hubmap.yaml --device cpu')
        os.chdir(self.yolodir)

        return
    

    def prepare_predicting_yolo(self):
        """ Prepares data for YOLO prediction on a inference set. """

        if self.mode == 'test':
            print(f"Creating bb annotations for wsis in '{self.src_dir}' folder:")
            get_wsi_bb(source_folder=self.src_dir , convert_to= 'yolo')
            print(f"Creating bb annotations for tiles in {self.src_dir} folder:")
            get_tiles_bb(folder = self.src_dir, shape_patch = self.tile_shape)
            print(f"Moving created tiles and annotations to their folders: ")
            move_tiles(src_folder = self.src_dir, dst_folder =self.dst_dir, mode = 'test')
            print(f'generating slides for: {self.slides_fns}')
            dst = os.path.join(self.dst_dir, 'predictions')
            get_tiles_masks(self.src_dir, out_folder = dst)
            utils_yolo.edit_yaml(root = self.dst_dir)
        elif self.mode == 'train': # i.e. if predictions is to generate images for unet to be trained
            self.prepare_training_yolo()

        return


    def predict_yolo(self, dir):
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = self.yolo_weights

        print(f'Loading weights from {weights_dir}')
        os.system(f'python detect.py --source {dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --conf-thres {self.conf_thres} --save-txt --class 0')
        os.chdir(self.yolodir)

        return
    

    def prepare_unet(self, tiles_imgs_folder:str = False):

        # 1- detection with the YOLO on test folder
        # self.predict_yolo(dir = self.train_dir.replace('wsis', 'tiles'))
        # 2 - get results 

        if self.mode == 'test':
            if tiles_imgs_folder is False:
                raise TypeError("''tiles_imgs_folder' can't be False if mode is 'test'")
            txt_folder = utils_yolo.get_last_detect()
            save_imgs_folder = tiles_imgs_folder.replace('tiles', 'crops')
            print(f'Cropping images from: {txt_folder} and {tiles_imgs_folder}')
            print(f'Saving images in {save_imgs_folder}')
            crop_obj(txt_folder= txt_folder, 
                     tiles_imgs_folder= tiles_imgs_folder, 
                     save_imgs_folder = save_imgs_folder,
                     crop_shape = 512)

        # 4 - create images and masks 
        if self.mode == 'train':
            print("Preparing for YOLO prediction: ")
            self.prepare_predicting_yolo()
            dir_names = ['train', 'val', 'test']
            dir_paths = [self.train_dir, self.val_dir, self.test_dir]
            tiles_dirs = [ os.path.join(dir, f'model_{name}', 'images') for dir, name in zip(dir_paths, dir_names)]
            for dirname, dir in zip(dir_names, tiles_dirs):
                print(f"Predicting with YOLO on {dir}")
                self.predict_yolo(dir = dir)
                print(f"Getting last detected images path: ")
                txt_folder = utils_yolo.get_last_detect()
                save_imgs_folder = os.path.join(dir.replace('images', 'unet'), 'images')
                print(f'Cropping images from: {txt_folder} and {dir}')
                crop_obj(txt_folder= txt_folder, 
                        tiles_imgs_folder= dir, 
                        save_imgs_folder = save_imgs_folder,
                        crop_shape = 512)

        return
    

    def infere_testset(self):
        """ Segments images using pretrained U-Net on test folder. """

        if self.system == 'windows':
            raise NotImplemented('test_unet method not yet implemented for windows. need to define image_folder path for that case')
        image_folder = '/Users/marco/hubmap/training/val/model_val/unet' 

        preds_folder = pu.predict(image_folder, 
                                  plot = True, 
                                  save_plot_every= 2, 
                                  path_to_exps='/Users/marco/hubmap/unet/lightning_logs',
                                  classes = 1)

        return
    

    def test_unet(self, version_n: str = None, dataset = 'test'):
        """ version_n: str = number of version folder where the model and hparams is stored. """

        model = GlomModel(arch = 'unet',
                          encoder_name='resnet34', 
                          encoder_weights='imagenet',
                          in_channels = 3,
                          out_classes = self.unet_classes)

        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs' if self.system == 'mac' else r'C:\marco\biopsies\zaneta\lightning_logs'

        if version_n is None:
            weights_path, hparams_path = utils_yolo.get_last_model(path_to_exps=path_to_exps)
        else:
            version_fold = os.path.join(path_to_exps, f'version_{version_n}')
            weights_path = os.listdir(os.path.join(version_fold, 'checkpoints'))
            weights_path = [os.path.join(version_fold, 'checkpoints', file) for file in weights_path if '.ckpt' in file][0]
            hparams_path = os.path.join(version_fold, 'hparams.yaml')

        print(f"Loading model from '{weights_path}'")
    
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        train_img_dir = os.path.join(self.train_dir, 'model_train', 'unet', 'images')
        val_img_dir = os.path.join(self.val_dir, 'model_val', 'unet', 'images',)
        test_img_dir = os.path.join(self.test_dir, 'model_test', 'unet', 'images')

        _, val_loader, test_loader = get_loaders(train_img_dir, val_img_dir, test_img_dir, resize = self.unet_resize, classes = self.unet_classes)

        model = model.load_from_checkpoint(checkpoint_path=weights_path, hparams_file=hparams_path)

        if self.system == 'mac':
            trainer = pl.Trainer(accelerator='mps', devices = 1)
        elif self.system == 'windows':
            trainer = pl.Trainer( max_epochs = self.unet_epochs )

        dataloader = val_loader if dataset == 'val' else test_loader
        trainer.validate(model = model, dataloaders=dataloader, ckpt_path=weights_path)

        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

        return


    def predict_unet(self, 
                    image_folder: str,
                    reconstruct: bool = False,
                    coords_folder: str = None, 
                    crops_folder: str = None,
                    reconstructed_tiles_folder: str = None):
        """ Segments images using pretrained U-Net on test folder. """

        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs' if self.system == 'mac' else r'C:\marco\biopsies\zaneta\lightning_logs'
        preds_folder = pu.predict(image_folder, path_to_exps=path_to_exps, plot = False, save_plot_every= 2, classes = 1)
        # reconstruct(preds_folder = preds_folder,
        #             coords_folder = coords_folder ,
        #             crops_folder = preds_folder,
        #             reconstructed_tiles_folder= reconstructed_tiles_folder,
        #             plot = True )


        return
    

    def train_unet(self):
        """  Trains the U-Net model.  """

        if self.system == 'mac':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        self.unet_hparams = {'arch' : 'unet',
                            'encoder_name': 'resnet34', 
                            'encoder_weights': 'imagenet', 
                            'in_channels' : 3,
                            'out_classes': 1,
                            'activation' : None}

        model = GlomModel(
            arch = self.unet_hparams['arch'],
            encoder_name = self.unet_hparams['encoder_name'], 
            encoder_weights = self.unet_hparams['encoder_weights'], 
            in_channels = self.unet_hparams['in_channels'],
            out_classes = self.unet_hparams['out_classes'],
            activation = self.unet_hparams['activation'])

        # create_dirs()
        train_img_dir = os.path.join(self.train_dir, 'model_train', 'unet', 'images')
        val_img_dir = os.path.join(self.val_dir, 'model_val', 'unet', 'images',)
        test_img_dir = os.path.join(self.test_dir, 'model_test', 'unet', 'images')

        train_loader, val_loader, _ = get_loaders(train_img_dir, val_img_dir, test_img_dir, resize = self.unet_resize, classes = 1)

        os.chdir(self.unet_weights_save_path)
        if self.system == 'mac':
            trainer = pl.Trainer( max_epochs = self.unet_epochs, 
                                  accelerator='mps', 
                                  devices = 1, 
                                  )
        elif self.system == 'windows':
            trainer = pl.Trainer( max_epochs = self.unet_epochs, )

        trainer.fit(model,
                    train_dataloaders = train_loader, 
                    val_dataloaders = val_loader)

        os.chdir(self.code_dir)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs' if self.system == 'mac' else r'C:\marco\biopsies\zaneta\lightning_logs'
        last, hparams_file = utils_unet.get_last_model(path_to_exps= path_to_exps)
        utils_unet.write_hparams_yaml(hparams_file= hparams_file, hparams = self.unet_hparams)

        return
    

    def train(self):
        """ Runs the whole pipeline end2end, i.e. runs both YOLO and U-Net. """

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


    def predict(self):
        """   Predicts using the entire pipeline.   """

        if self.mode == 'test':
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
        
\

if __name__ == '__main__':
    
    def get_folders(system = 'windows'):
        if system == 'windows':
            src_folder = r'C:\marco\biopsies\hubmap\slides'
            dst_folder = r'C:\marco\biopsies\hubmap'
        elif system == 'mac':
            src_folder = '/Users/marco/glomseg-share'        
            dst_folder = '/Users/marco/datasets/muw_exps'
        
        return src_folder, dst_folder

    src_folder, dst_folder = get_folders('mac')

    manager = Manager(src_folder = src_folder,
                      dst_folder = dst_folder,
                      ratio = [0.7, 0.15, 0.15], 
                      mode = 'train',
                      system= 'mac',
                      unet_epochs= 10,
                      unet_classes= 1,
                      yolo_epochs=50, 
                      yolo_val_augment=False)


    # manager.test_unet(version_n = '1', dataset='val')
    manager.test_yolo(conf_thres=0.43)
