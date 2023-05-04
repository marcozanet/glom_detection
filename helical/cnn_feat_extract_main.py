import torch, torchvision
from torchvision import models
from torch import nn
from cnn_feat_extract_funcs import prepare_data, feature_extraction
from utils import get_config_params


def extract_cnn_features():

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # params
    PARAMS = get_config_params('cnn_feature_extractor')
    cnn_root_fold = PARAMS['cnn_root_fold']
    map_classes = PARAMS['map_classes']
    num_workers = PARAMS['num_workers']
    vgg_weights_path = PARAMS['vgg_weights_path']
    cnn_weights_path = PARAMS['cnn_weights_path']
    yolo_root = PARAMS['yolo_root']
    exp_folds = PARAMS['exp_folds']
    resize_crops = PARAMS['resize_crops']
    batch = PARAMS['batch']

    # Load the pretrained model from pytorch
    vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT') # pretrained on COCO (?)
    vgg16.load_state_dict(torch.load(vgg_weights_path))

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Delete last layer from vgg and add classification layer:
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, len(map_classes))]) # Add our layer; num classes + 1 because also false-positives = bg
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    # Load pretrained weights from CNN
    model = vgg16
    model.load_state_dict(torch.load(cnn_weights_path))

    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False

    # Delete last layer from vgg (feat extraction):
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    model.classifier = nn.Sequential(*features) # Replace the model classifier

    # torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')
    dataloader = prepare_data(cnn_root_fold=cnn_root_fold, map_classes=map_classes, batch=batch, 
                              num_workers=num_workers, exp_folds=exp_folds, yolo_root=yolo_root, resize_crops=resize_crops)
    features = feature_extraction(model, dataloader, cnn_root_fold=cnn_root_fold)

    return 


if __name__ == '__main__':
    extract_cnn_features()
