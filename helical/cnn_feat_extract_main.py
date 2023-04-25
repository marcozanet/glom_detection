import torch, torchvision
from torchvision import models
from torch import nn
from cnn_feat_extract_funcs import prepare_data, feature_extraction


def extract_features():

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # params
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn_root_fold = '/Users/marco/helical_tests/test_featureextractor/test_fullpipeline'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
    feat_fold = '/Users/marco/Downloads/extracted'
    root_dir = '/Users/marco/helical_tests/test_cnn_processor/test_crossvalidation/feat_extract'
    batch = 1
    num_workers = 0
    vgg_weights_path = '/Users/marco/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth'
    cnn_weights_path = '/Users/marco/helical_tests/cnn_model/exps/VGG16_v2-OCT_Retina_half_dataset.pt'
    yolo_root = '/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/detection'
    exp_folds = ['/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/exp30',
                '/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/exp31_fake',
                 '/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/exp32_fake' ]
    task = 'detection'
    resize_crops = True

    # Load the pretrained model from pytorch
    vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
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
    extract_features()
