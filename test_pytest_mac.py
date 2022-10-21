import unet_utils
import torch


def test_get_last_model():
    
    path_to_exps= '/Users/marco/hubmap/unet/lightning_logs'
    last, hparams_file = unet_utils.get_last_model(path_to_exps)
    assert '.ckpt' in last, f"'{last}' not in .ckpt format"
    assert '.yaml' in hparams_file, f"'{hparams_file}' not in .yaml format"

    return


def test_write_hparams_yaml():

    hparams_file = '/Users/marco/yolo/lightning_logs/version_0/hparams.yaml'
    hparams = {'arch' : 'unet',
    'encoder_name': 'resnet34', 
    'encoder_weights': 'imagenet', 
    'in_channels' : 3,
    'out_classes': 3,
    'activation' : None}
    unet_utils.write_hparams_yaml(hparams_file = hparams_file, hparams = hparams)

    return


def test_binary_glom_dataset():

    train_img_dir = '/Users/marco/zaneta-tiles-pos0_02/train/images'
    resize = 128
    classes = 1

    val_img_dir = train_img_dir.replace('train', 'val')
    test_img_dir =  train_img_dir.replace('train', 'test')
    trainset = unet_utils.GlomDataset(img_dir=train_img_dir, resize = resize, classes = classes)
    valset = unet_utils.GlomDataset(img_dir=val_img_dir, resize = resize, classes = classes)
    testset = unet_utils.GlomDataset(img_dir=test_img_dir, resize = resize, classes = classes)

    print(f"Train size: {len(trainset)} images.")
    print(f"Valid size: {len(valset)} images." )
    print(f"Test size: {len(testset)} images.")

    data = trainset[0]
    image = data['image']
    mask = data['mask']


    assert len(image.shape) == 3, f"Image {image.shape} should have 3 dims."
    assert len(mask.shape) == 3, f"Image {mask.shape} should have 3 dims."
    assert image.shape[1] == (resize) and image.shape[2] == resize, f"Image shape is {image.shape}, but should be shape (C, {resize},{resize})"
    assert mask.shape[1] == (resize) and mask.shape[2] == resize, f"Mask shape is {mask.shape}, but should be shape (C, {resize},{resize})"
    assert mask.shape[0] == 1, f"Mask has shape {mask.shape}, but 1st channel should be = 1 for both binary and multiclass cases."
    assert isinstance(image, torch.Tensor), f"Image type = {type(image)}, but should be torch.Tensor instead. "
    assert isinstance(mask, torch.Tensor), f"Mask type = {type(mask)}, but should be torch.Tensor instead. "

    return


def test_multiclass_glom_dataset():

    train_img_dir = '/Users/marco/zaneta-tiles-pos0_02/train/images'
    resize = 128
    classes = 3

    val_img_dir = train_img_dir.replace('train', 'val')
    test_img_dir =  train_img_dir.replace('train', 'test')
    trainset = unet_utils.GlomDataset(img_dir=train_img_dir, resize = resize, classes = classes)
    valset = unet_utils.GlomDataset(img_dir=val_img_dir, resize = resize, classes = classes)
    testset = unet_utils.GlomDataset(img_dir=test_img_dir, resize = resize, classes = classes)

    print(f"Train size: {len(trainset)} images.")
    print(f"Valid size: {len(valset)} images." )
    print(f"Test size: {len(testset)} images.")

    data = trainset[0]
    image = data['image']
    mask = data['mask']


    assert len(image.shape) == 3, f"Image {image.shape} should have 3 dims."
    assert len(mask.shape) == 3, f"Image {mask.shape} should have 3 dims."
    assert image.shape[1] == (resize) and image.shape[2] == resize, f"Image shape is {image.shape}, but should be shape (C, {resize},{resize})"
    assert mask.shape[1] == (resize) and mask.shape[2] == resize, f"Mask shape is {mask.shape}, but should be shape (C, {resize},{resize})"
    assert mask.shape[0] == 1, f"Mask has shape {mask.shape}, but 1st channel should be = 1 for both binary and multiclass cases."
    assert isinstance(image, torch.Tensor), f"Image type = {type(image)}, but should be torch.Tensor instead. "
    assert isinstance(mask, torch.Tensor), f"Mask type = {type(mask)}, but should be torch.Tensor instead. "

    return

def test_pred_mask2colors():

    n_colors = 3
    pred_mask = torch.randint(low = 0, high = n_colors, size = (1024, 1024) )
    colored_mask = unet_utils.pred_mask2colors(pred_mask = pred_mask, n_colors=2)
    assert colored_mask.shape == pred_mask.shape, f"colored_mask {colored_mask} shape should be equal to pred_mask shape {pred_mask}. "
    assert len(colored_mask.unique()) == n_colors, f"Output colors: {len(colored_mask.unique())}, but input colors= {n_colors}"

    n_colors = 2
    pred_mask = torch.randint(low = 0, high = n_colors, size = (512, 512) )
    colored_mask = unet_utils.pred_mask2colors(pred_mask = pred_mask, n_colors=2)
    assert colored_mask.shape == pred_mask.shape, f"colored_mask {colored_mask} shape should be equal to pred_mask shape {pred_mask}. "
    assert len(colored_mask.unique()) == n_colors, f"Output colors: {len(colored_mask.unique())}, but input colors= {n_colors}"
    return

if __name__ == '__main__':
    test_pred_mask2colors()