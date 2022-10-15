
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from unet_utils import GlomDataset, IoULoss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt



# create experiment dir for results
def create_dirs():

    cwd = os.getcwd()
    exp_dir = os.path.join(cwd, 'unet_runs')
    if not os.path.isdir(exp_dir):
        exp_dir = os.path.join(exp_dir, 'exp_0')
        os.makedirs(exp_dir)
    else:
        exps = [folder for folder in os.listdir(exp_dir) if 'exp' in folder]
        if len(exps) == 0:
            exp_dir = os.path.join(exp_dir, 'exp_0')
            os.makedirs(exp_dir)
        else:
            exps = [folder for folder in os.listdir(exp_dir) if 'exp' in folder]
            run_nums = [int(os.path.split(folder)[1].split('_')[1].split('.')[0]) for folder in exps]
            last_run_num = np.array(run_nums).max()
            exp_dir = os.path.join(exp_dir, f'exp_{last_run_num+1}' )
            os.makedirs(exp_dir)
    print(f'Experiment results will be saved at {exp_dir}')

    return


def get_loaders(train_img_dir, val_img_dir, test_img_dir, batch = 2, num_workers = 8, resize = False, classes = 2):

    # get train, val, test set
    trainset = GlomDataset(img_dir=train_img_dir, resize = resize, classes = 3)
    valset = GlomDataset(img_dir=val_img_dir, resize = resize, classes = 3)
    testset = GlomDataset(img_dir=test_img_dir, resize = resize, classes = 3)

    # It is a good practice to check datasets don`t intersects with each other
    # assert set(trainset.imgs_fn).isdisjoint(set(valset.imgs_fn))
    # assert set(valset.imgs_fn).isdisjoint(set(testset.imgs_fn))
    # assert set(trainset.imgs_fn).isdisjoint(set(testset.imgs_fn))

    print(f"Train size: {len(trainset)} images.")
    print(f"Valid size: {len(valset)} images." )
    print(f"Test size: {len(testset)} images.")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader


# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=4,                 # define number of output labels
# )

class GlomModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        if type(image) == dict:
            image = image['image']
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)



if __name__ == '__main__':
    # TODO PRINT A COUPLE IMAGES 
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.chdir('/Users/marco/hubmap/unet')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(device)
    train_img_dir = '/Users/marco/zaneta-tiles-pos0_02/train/images'
    val_img_dir = train_img_dir.replace('train', 'val')
    test_img_dir =  train_img_dir.replace('train', 'test')
    # create_dirs()
    train_dataloader, val_dataloader, _ = get_loaders(train_img_dir, 
                                                     val_img_dir, 
                                                     test_img_dir, 
                                                     classes = 3, 
                                                     batch = 2)
    trainer = pl.Trainer(max_epochs=10, 
                        accelerator='mps', 
                        weights_save_path= '/Users/marco/hubmap/unet'
                        )
    model = GlomModel(
        arch = 'unet',
        encoder_name='resnet34', 
        encoder_weights='imagenet',
        in_channels = 3,
        out_classes = 3,
        # aux_params = aux_params
    )
    # print(model)
    # preprocess_input = get_preprocessing_fn('resnet18', pretrained
    #  ='imagenet')
    trainer.fit(model,
                train_dataloaders = train_dataloader, 
                val_dataloaders = val_dataloader)




