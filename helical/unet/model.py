
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl


class GlomModel(pl.LightningModule):

    def __init__(self, 
                arch, 
                encoder_name, 
                in_channels, 
                out_classes, 
                activation, 
                **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, activation = activation, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.dice.DiceLoss(mode = 'multiclass', from_logits=True, log_loss = False)


    def forward(self, x):
        """ Forward method. """

        # normalize image here
        if type(x) == dict: # on the first round
            image = x['image']
            print(f"\nimage: {image.shape}")
            print(f"mean: {self.mean.shape}")
            print(f"std: {self.std.shape}")
            image = (image - self.mean) / self.std

        x = self.model(x)

        return x


    def shared_step(self, batch, stage):
        
        image = batch["image"]

        assert image.ndim == 4

        h, w = image.shape[2:]
        # assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        assert mask.ndim == 4, f"Mask ndim = {mask.ndim}, but should be 4 [batch_size, num_classes, height, width]."
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.softmax(dim = 1) #softmax(dim = 1)
        pred_mask = prob_mask.argmax(dim = 1, keepdim = True)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass", num_classes = 3)

        return {"loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn}


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
