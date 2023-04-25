import torch, torchvision
from torchvision import models
from torch import nn
from cnn_train_func import train_model, eval_model
from cnn_loaders import CNNDataLoaders
import os
from cnn_crossvalidation import CNN_KCrossValidation
from datetime import datetime
from cnn_train_func import prepare_data

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# params
device='cuda:0' if torch.cuda.is_available() else 'cpu'
yolo_data_root = '/Users/marco/helical_tests/test_yolo_detect_train_muw_sfog/detection'
cnn_data_root = '/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/cnn_dataset'
map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
yolo_exp_folds = ['/Users/marco/helical_tests/test_featureextractor/test_fullpipeline/exp30']
lr = 10e-2
k_tot = 4
k_i = 0
batch = 5
epochs = 10
num_workers = 0
weights_path = '/Users/marco/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth'
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
cnn_exp_fold = '/Users/marco/helical_tests/cnn_model/exps'
weights_save_fold = cnn_exp_fold +f"_{dt_string}"
dataset = 'muw'
task = 'detection'
resize_crops = True

# prepare cnn dataset: 
cnn_data_fold = prepare_data(cnn_root_fold=cnn_data_root, map_classes=map_classes, batch=batch, resize_crops=resize_crops,
                            num_workers=num_workers, yolo_root=yolo_data_root, yolo_exp_folds=yolo_exp_folds)
crossvalidator = CNN_KCrossValidation(data_root=cnn_data_fold, k=k_tot, dataset = dataset, task=task)
crossvalidator._change_kfold(fold_i=k_i)

# raise NotImplementedError()

# processor
dataloader_cls = CNNDataLoaders(root_dir=cnn_data_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
dataloaders = dataloader_cls()

# raise NotImplementedError()
# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
vgg16.load_state_dict(torch.load(weights_path))
# print(vgg16)

# raise NotImplementedError()
# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(map_classes))]) # Add our layer; num classes + 1 because also false-positives = bg
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# train model 
model = vgg16
criterion = nn.BCEWithLogitsLoss()
optimizer_ft = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
vgg16 = train_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, 
                    criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epochs)
os.makedirs(weights_save_fold)
torch.save(vgg16.state_dict(), os.path.join(weights_save_fold, 'VGG16_v2-OCT_Retina_half_dataset.pt'))
eval_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, criterion=criterion)
