import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import utils_yolo, utils_unet, utils_manager
from typing import List
from model import GlomModel
from dataloader import get_loaders
import pytorch_lightning as pl
import predict_unet as pu
from processor_tile import TileProcessor
from processor_wsi import WSI_Processor
import time
import datetime
from loggers import get_logger
from typing import Literal, List


class Manager():

    def __init__(self, 
                src_folder:str,
                dst_folder: str,
                data_tiled: bool,
                mode: Literal['train', 'test', 'detect'],
                task: str,
                model: str,
                convert_from: Literal['json_wsi_mask', 'jsonliketxt_wsi_mask'], 
                convert_to: Literal['json_wsi_bboxes', 'txt_wsi_bboxes'],  
                slide_format: Literal['tiff', 'tif'],
                system: Literal['windows', 'mac'],
                ratio: float = [0.7, 0.15, 0.15], 
                tile_shape = (4096, 4096),
                yolo_tiles = 512,
                yolo_batch = 8,
                yolo_epochs = 3,
                conf_thres = None,
                unet_classes = 3,
                unet_tiles = 512,
                unet_epochs = 20,
                unet_batch = 8,
                step: int = None,
                unet_resize = False,
                unet_weights_save_path = '/Users/marco/hubmap/unet/',
                yolo_weights = False,
                yolo_val_augment = False, 
                yolo_infere_augment = False,
                unet_ratio: List[float] = [0.7, 0.15, 0.15],
                unet_RGBmapping: dict = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2},
                percentile:int = 90,
                empty_perc: float = 0.1
                ) -> None: 

        self.log = get_logger()
        assert isinstance(conf_thres, float) or conf_thres is None, TypeError(f"conf_thres is {type(conf_thres)}, but should be either None or float.")
        assert model in ['yolo', 'unet'], f"'model' should be either 'yolo' or 'unet'."
        assert task in ['segmentation', 'detection'], f"'task' should be either 'segmentation' or 'detection'."
        assert isinstance(data_tiled, bool), f"'data_tiled' should be boolean."
        assert isinstance(step, int) or step is None, f"'step' should be either None or int."
        assert isinstance(empty_perc, float), f"'empty_perc' should be a float between 0 and 1."
        assert 0<=empty_perc<=1, f"'empty_perc' should be a float between 0 and 1."
        assert mode in ['train', 'test', 'detect'], f"'mode' should be one of ['train', 'test', 'detect']."
        assert slide_format in ['tiff', 'tif'], f"'slide_format' should be either 'tiff' or 'tif' format."


        self.src_dir = src_folder
        self.dst_dir = dst_folder
        self.root = dst_folder
        self.task = task
        self.model = model
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
        self.yolo_infere_augment = yolo_infere_augment
        self.conf_thres = conf_thres
        self.tile_shape = tile_shape
        self.percentile = percentile
        self.unet_batch = unet_batch
        self.unet_RGBmapping = unet_RGBmapping
        self.data_tiled = data_tiled
        self.step = step
        self.empty_perc = empty_perc
        self.mode = mode
        self.slide_format = slide_format
        self.convert_from = convert_from
        self.convert_to = convert_to


        if self.system == 'windows':
            self.yolov5dir = 'C:\marco\yolov5'
            self.yolodir = 'C:\marco\code\glom_detection'
            self.code_dir = ''
            # raise Exception('define weights path for unet')
        elif self.system == 'mac':
            self.yolodir = '/Users/marco/yolo'
            self.yolov5dir = '/Users/marco/yolov5'
            self.unet_weights_save_path = unet_weights_save_path
            self.code_dir = os.path.join(self.yolodir, 'code')


        return
    
    def _prepare_data(self):

        assert not (self.step is None and self.data_tiled is False), f"If data is not tiled ('data_tiled' is False), 'step' should be provided."
        assert not (self.empty_perc is None and self.data_tiled is False), f"If data is not tiled ('data_tiled' is False), '' should be provided."

        if self.data_tiled is True:
            processor = TileProcessor(src_root=src_folder,
                                      dst_root=dst_folder,
                                      task=self.task,
                                      ratio=self.ratio)
        else:
            processor = WSI_Processor(src_root= self.src_dir, 
                                      dst_root=self.dst_dir,
                                      task = self.task, 
                                      ratio=self.ratio, 
                                      slide_format = self.slide_format,
                                      convert_from = self.convert_from,
                                      convert_to = self.convert_to,
                                      step = self.step,
                                      empty_perc=self.empty_perc)
        
        self.traindir, self.valdir, self.testdir = processor()

        return
    

    def _prepare_training_yolo_detection(self) -> dict:
        """ Prepares YOLO detection training . """

        # prepare YAML file:
        yolo_classes = {0: 'healthy', 1: 'unhealthy'}
        utils_manager.edit_yaml(root = self.dst_dir, task=self.task, system = self.system, classes = yolo_classes )
        self.log.info(f"Prepared training YOLO ✅ .")

        return yolo_classes


    def _prepare_training_yolo_segmentation(self) -> dict:
        """ Prepares YOLO segmentation training. """

        # prepare YAML file:
        yolo_classes = {0: 'healthy', 1: 'unhealthy'}
        print(self.dst_dir)
        utils_manager.edit_yaml(root = self.dst_dir, task=self.task, system = self.system, classes = yolo_classes )
        self.log.info(f"Prepared training YOLO ✅ .")

        return yolo_classes


    def _train_yolo_detection(self) -> None:
        """   Runs the YOLO detection model. """
        
        # 1) prepare training:
        start_time = time.time()
        yolo_classes = self._prepare_training_yolo_detection()

        # 2) train:
        self.log.info(f"Start training YOLO: ⏳")
        os.chdir(self.yolov5dir)
        os.system(f' python train.py --img {self.yolo_tiles} --batch {self.yolo_batch} --epochs {self.yolo_epochs} --data hubmap.yaml --weights yolov5s.pt')
        os.chdir(self.yolodir)

        # 3) save:
        end_time = time.time()
        train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        otherinfo_yolo = {'data': self.src_dir, 'classes': yolo_classes, 'epochs': self.yolo_epochs, 'duration': train_yolo_duration}
        utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = '/Users/marco/yolov5/runs/train')
        self.log.info(f"Training YOLO done ✅ . Training duration: {train_yolo_duration}")

        return  


    def _train_yolo_segmentation(self) -> None:
        """   Runs the YOLO segmentation model. """

        # 1) prepare:
        start_time = time.time()
        yolo_classes = self._prepare_training_yolo_segmentation()

        # 2) train:
        self.log.info(f"Start training segmentation YOLO: ⏳")
        os.chdir(self.yolov5dir)
        os.system(f' python train.py --img {self.yolo_tiles} --batch {self.yolo_batch} --epochs {self.yolo_epochs} --data hubmap.yaml --weights yolov5s.pt')
        os.chdir(self.yolodir)

        # 3) save:
        end_time = time.time()
        train_yolo_duration = datetime.timedelta(seconds = end_time - start_time)
        otherinfo_yolo = {'data': self.src_dir, 'classes': yolo_classes, 'epochs': self.yolo_epochs, 'duration': train_yolo_duration}
        utils_manager.write_YOLO_txt(otherinfo_yolo, root_exps = '/Users/marco/yolov5/runs-seg/train')
        self.log.info(f"Training segmentation YOLO done ✅ . Training duration: {train_yolo_duration}")

        return 


    def _prepare_testing_yolo_detection(self) -> str:
        """ Prepares testing with YOLO. """

        # get model
        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = self.yolo_weights

        self.log.info(f"Prepared YOLO testing ✅ .")

        return weights_dir

    
    def _test_yolo_detection(self) -> None:
        """ Tests YOLO on the test set and returns performance metrics. """
        
        # 1) prepare testing:
        weights_dir = self._prepare_testing_yolo_detection()

        # 2) define command:
        command = f'python val.py --task test --weights {weights_dir} --data data/hubmap.yaml --device cpu'
        if self.yolo_val_augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf_thres {self.conf_thres}" 

        # 3) test (e.g. validate):
        self.log.info(f"Start testing YOLO: ⏳")
        os.system(command)
        os.chdir(self.yolodir)
        self.log.info(f"Testing YOLO done ✅ .")

        return
    


    def _prepare_infering_yolo_detection(self) -> str:
        """ Prepares testing with YOLO. """

        # get model:
        os.chdir(self.yolov5dir)
        if self.yolo_weights is False:
            weights_dir = utils_yolo.get_last_weights()
        else:
            weights_dir = self.yolo_weights

        self.log.info(f"Prepared YOLO for inference ✅ .")

        return weights_dir


    def _infere_yolo_detection(self, images_dir: str) -> None:
        """ Predicts bounding boxes for images in dir and outputs txt labels for those boxes. """

        assert os.path.isdir(images_dir), ValueError(f"'images_dir': {images_dir} is not a valid dirpath.")

        # 1) prepare inference:
        weights_dir = self._prepare_infering_yolo_detection()

        # 2) define command:
        command = f'python detect.py --source {images_dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --save-txt '
        if self.yolo_infere_augment is True:
            command += " --augment"
        if self.conf_thres is not None:
            command += f" --conf_thres {self.conf_thres}" 

        # 3) infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.yolodir)
        self.log.info(f"Inference YOLO done ✅ .")

        return
    
    def _infere_yolo_segmentation(self):
        raise NotImplementedError()
        return

    
    def _infere_for_UNet(self) -> None:
        """ Applies inference with YOLO on train, val, test sets. """

        # 1) prepare inference:
        weights_dir = self._prepare_infering_yolo_detection()

        # 2) infere on train, val, test sets:
        dirs = [self.train_dir, self.val_dir, self.test_dir]
        dirs = [os.path.join(dir, 'images') for dir in dirs]
        dirnames = ['train', 'val', 'test']
        for dirname, image_dir in zip(dirnames, dirs):
            command = f'python detect.py --source {image_dir} --weights {weights_dir} --data data/hubmap.yaml --device cpu --save-txt '
            if self.yolo_infere_augment is True:
                command += " --augment"
            self.log.info(f"Start inference YOLO on {dirname}: ⏳")
            os.system(command)
            self.log.info(f"Inference YOLO done on {dirname} ✅ .")

        os.chdir(self.yolodir)

        return


    def _prepare_unet(self) -> tuple:
        """ Inferes using YOLO, crops inferred images and splits them into train, val, test sets. """

        # 1) infere using YOLO on train, val, test:
        # self._infere_for_UNet()

        # 2) crop inferred images
        # save_folder = os.path.join(self.dst_dir, 'segmentation/images')
        # cropper = Cropper(root = self.dst_dir, 
        #                   save_folder = save_folder,
        #                   image_shape = self.tile_shape,
        #                   percentile = self.percentile)
        # cropper()

        if self.system == 'mac':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 1) creating model:
        self.unet_hparams = {'arch' : 'unet',
                            'encoder_name': 'resnet34', 
                            'encoder_weights': 'imagenet', 
                            'in_channels' : 3,
                            'out_classes': self.unet_classes,
                            'activation' : None}

        model = GlomModel(arch = self.unet_hparams['arch'],
                          encoder_name = self.unet_hparams['encoder_name'], 
                          in_channels= self.unet_hparams['in_channels'],
                          encoder_weights = self.unet_hparams['encoder_weights'], 
                          out_classes = self.unet_hparams['out_classes'],
                          activation = self.unet_hparams['activation'])

        # 2) get loaders:
        train_loader, val_loader, _ = get_loaders(train_img_dir = os.path.join(self.train_dir, 'images'), 
                                                  val_img_dir= os.path.join(self.val_dir, 'images'),
                                                  test_img_dir = os.path.join(self.test_dir, 'images'), 
                                                  resize = self.unet_resize, 
                                                  batch = self.unet_batch,
                                                  classes = self.unet_hparams['out_classes'])

        # 3) load pre-trained weights and architecture:
        os.chdir(self.unet_weights_save_path)
        print(f"Unet weights: {self.unet_weights_save_path}")
        if self.system == 'mac':
            trainer = pl.Trainer( max_epochs = self.unet_epochs, 
                                  accelerator='mps', 
                                  devices = 1)
        elif self.system == 'windows':
            trainer = pl.Trainer( max_epochs = self.unet_epochs )
        
        return  model, trainer, train_loader, val_loader


    def train_yolo_segmentation(self) -> None:
        """  Trains the YOLO segmentation model.  """

        self.log.info(f"Preparing U-Net: ⏳")
        model, trainer, train_loader, val_loader = self._prepare_unet()
        self.log.info(f"Preparation U-Net done ✅ .")

        # 4) train model:
        self.log.info(f"Training U-Net: ⏳")
        trainer.fit(model,
                    train_dataloaders = train_loader, 
                    val_dataloaders = val_loader)
        self.log.info(f"Training U-Net done ✅ .")

        # 5) save:
        os.chdir(self.code_dir)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs' if self.system == 'mac' else r'C:\marco\biopsies\zaneta\lightning_logs'
        last, hparams_file = utils_unet.get_last_model(path_to_exps= path_to_exps)
        utils_unet.write_hparams_yaml(hparams_file= hparams_file, hparams = self.unet_hparams)

        return
    
    def test_yolo_segmentation(self):

        raise NotImplementedError()

        return

#################################### IMPLEMENTING ####################################

    def infere_unet(self) -> None:
        """  Inferes with U-Net model.  """

        self.log.info(f"Preparing U-Net: ⏳")
        model, trainer, train_loader, val_loader = self._prepare_unet()
        self.log.info(f"Preparation U-Net done ✅ .")

        # 4) train model:
        self.log.info(f"Training U-Net: ⏳")
        trainer.fit(model,
                    train_dataloaders = train_loader, 
                    val_dataloaders = val_loader)
        self.log.info(f"Training U-Net done ✅ .")

        # 5) save:
        os.chdir(self.code_dir)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        path_to_exps = '/Users/marco/hubmap/unet/lightning_logs' if self.system == 'mac' else r'C:\marco\biopsies\zaneta\lightning_logs'
        last, hparams_file = utils_unet.get_last_model(path_to_exps= path_to_exps)
        utils_unet.write_hparams_yaml(hparams_file= hparams_file, hparams = self.unet_hparams)

        return

#################################### NOT IMPLEMENTED ####################################

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

        _, val_loader, test_loader = get_loaders(train_img_dir, 
                                                val_img_dir, 
                                                test_img_dir, 
                                                resize = self.unet_resize, 
                                                classes = self.unet_classes,
                                                mapping = self.unet_RGBmapping)

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
    

    

    def train(self):
        """ Runs the whole pipeline end2end, i.e. runs both YOLO and U-Net. """

        self._train_yolo() 

        return
    
    def parse_args(self):
        raise NotImplementedError()
        return

    def __call__(self) -> None:

        self._prepare_data()

        raise NotImplementedError()

        if self.mode == 'train':
            if self.task == 'detection':
                self._train_yolo_detection()
            else:
                self._train_yolo_segmentation()
        elif self.mode == 'test':
            if self.task == 'detection':
                self._test_yolo_detection()
            else:
                self.test_yolo_segmentation()
        else:
            if self.task == 'detection':
                self._infere_yolo_detection()
            else:
                self._infere_yolo_segmentation()


        return


    # def predict(self):
    #     """   Predicts using the entire pipeline.   """

    #     if self.mode == 'test':
    #         for slide in self.slides_fns:
    #             # print("Predicting with YOLO on {slide}:")
    #             # self.predict_yolo( dir = os.path.join(self.dst_dir, 'predictions', slide, 'tiles', 'images'))
    #             print(f"Preparing U-Net for {slide}:")
    #             self.prepare_unet(tiles_imgs_folder = os.path.join(self.dst_dir, 'predictions', f'{slide}', 'tiles', 'images'))
    #             print(f"Segmenting {slide} with U-Net.")
    #             self.predict_unet(image_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'images'),
    #                             coords_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'bb') ,
    #                             crops_folder = os.path.join(self.dst_dir, 'predictions', slide, 'crops', 'images'),
    #                             reconstructed_tiles_folder= os.path.join(self.dst_dir, 'predictions', slide, 'tiles', 'reconstruced'))
    
    #     return
        


if __name__ == '__main__':
    
    def get_folders(system = 'windows'):
        if system == 'windows':
            src_folder = r'C:\marco\biopsies\hubmap\slides'
            dst_folder = r'C:\marco\biopsies\hubmap'
        elif system == 'mac':
            # src_folder = '/Users/marco/glomseg-share'        
            # dst_folder = '/Users/marco/datasets/muw_exps'
            src_folder = '/Users/marco/Downloads/converted_test'
            dst_folder = '/Users/marco/Downloads/folder_random'

        
        return src_folder, dst_folder

    src_folder, dst_folder = get_folders('mac')
    # unet_RGBmapping =  {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2, (255, 255, 255): 3}
    
    manager = Manager(src_folder = src_folder,
                      dst_folder = dst_folder,
                      data_tiled=False,
                      slide_format= 'tif',
                      task = 'detection',
                      convert_from='jsonliketxt_wsi_mask', 
                      convert_to='txt_wsi_bboxes',   
                      mode = 'train',
                      model = 'yolo',
                      step = 1024,
                      system= 'mac',
                     )
    
    manager()

    