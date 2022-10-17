import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os 
from unet_utils import GlomDataset, get_last_model
import numpy as nps
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from tqdm import tqdm
from skimage import io
import numpy as np
from unet_runner import GlomModel, create_dirs


# testloader 
def get_loader(img_dir: str, classes = 3):
    testset = GlomDataset(img_dir=img_dir, classes = classes) # TODO TODO TODO FAR SI CHE SIA UN PARAMETRO 
    print(f"Test size: {len(testset)} images.")
    # n_cpu = os.cpu_count()
    test_dataloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    return test_dataloader

# aux_params=dict(
#     pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     activation='sigmoid',      # activation function, default is None
#     classes=4,                 # define number of output labels
# )

def load_model():
    """ Load last trained model. NB: TODO: yaml configuration file needs to be filled. """
    

    model = GlomModel(
        arch = 'unet',
        encoder_name='resnet34', 
        encoder_weights='imagenet',
        in_channels = 3,
        out_classes = 3,
        # aux_params = aux_params
    )
    model_path, hparams_path = get_last_model('/Users/marco/hubmap/unet/lightning_logs')
    model = model.load_from_checkpoint(model_path, 
                                       hparams_file=hparams_path)
    
    return model


def create_pred_dir(test_dir: str):
    """ Creates dir where preds are saved. """

    preds_folder = os.path.join(test_dir, 'preds', )
    print(f'creating {preds_folder}')
    if not os.path.isdir(preds_folder):
        os.makedirs(preds_folder)
    
    return preds_folder


def predict(test_folder: str, classes: int = 3, plot: bool = False, save_plot_every: int = 5):
    """ Uses the last trained model to predict all images within a folder. """

    # GPU
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(device)

    # set folders and load model
    preds_folder = create_pred_dir(test_folder)
    test_dataloader = get_loader(os.path.join(test_folder, 'images'), classes = classes)
    model = load_model()

    # predict on 
    img_num = 0
    for batch in tqdm(test_dataloader, desc = 'Batch'):

        with torch.no_grad():
            model.eval()
            logits = model(batch["image"])
        if classes <= 2:
            pr_masks = torch.sigmoid()
        elif classes >= 3:
            pr_masks = torch.softmax(logits, dim = 1)
            pr_masks = pr_masks.argmax(dim = 1)
            print(pr_masks.shape)


        # save preds:
        for i, (image, gt_mask, pr_mask, fname) in enumerate(zip(batch["image"], batch["mask"], pr_masks, batch["fname"])):
            
            img_num += 1
            i += 1
            print(f"shape: {pr_mask.shape}" )
            if classes <= 2:
                preds = pr_mask.numpy().squeeze()
                preds *= 255
                gt_mask.numpy().squeeze()
            elif classes == 3:
                col_map = {0: 0, 1:127, 2:255}
                preds = pr_mask.numpy()
                preds = preds * 127.5
                gt_mask = gt_mask.numpy().transpose(1, 2, 0)
                print(gt_mask.shape)
                print(preds.shape)
            else:
                raise NotImplementedError()
            
            print(f" saving {fname}")
            io.imsave(fname = os.path.join(preds_folder, fname ), arr = np.uint8(preds), check_contrast=False)
            
            # plot:
            if plot is True:
                
                fig = plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask) # just squeeze classes dim, because we have only one class
                plt.title("Ground truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(preds) # just squeeze classes dim, because we have only one class
                title = f"Prediction_{i}"
                plt.title(title)
                plt.axis("off")

                plt.show()

                if i % save_plot_every == 0:
                    fig.savefig(os.path.join(preds_folder, title + '.png'))

    print(f'Img num: {img_num}')

    return preds_folder




if __name__ == '__main__':
    test_folder = '/Users/marco/zaneta-tiles-pos0_02/test'
    predict(test_folder, classes = 3, plot = True)
