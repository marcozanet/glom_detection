import torch, torchvision
from torchvision import models
from torch import nn
from cnn_train_func import train_model, eval_model
from cnn_loaders import CNNDataLoaders
import os, sys
from cnn_crossvalidation import CNN_KCrossValidation
from datetime import datetime
from cnn_train_func import prepare_data
from utils import get_config_params

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

############        PARAMS      ############      
PARAMS = get_config_params('cnn_trainer')
yolo_data_root = PARAMS['yolo_data_root']
cnn_data_root = PARAMS['cnn_data_root']
map_classes = PARAMS['map_classes']
yolo_exp_folds = PARAMS['yolo_exp_folds']
lr = PARAMS['lr']
k_tot = PARAMS['k_tot']
k_i = PARAMS['k_i']
batch = PARAMS['batch']
epochs = PARAMS['epochs']
num_workers = PARAMS['num_workers']
weights_path = PARAMS['weights_path']
cnn_exp_fold = PARAMS['cnn_exp_fold']
dataset = PARAMS['dataset']
task = PARAMS['task']
resize_crops = PARAMS['resize_crops']
treat_as_single_class = PARAMS['treat_as_single_class']
device = PARAMS['device']
gpu = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = device if device == 'mps' else gpu
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
weights_save_fold = cnn_exp_fold +f"_{dt_string}"


# prepare cnn dataset: 
cnn_data_fold = prepare_data(cnn_root_fold=cnn_data_root, map_classes=map_classes, batch=batch,
                            resize_crops=resize_crops, num_workers=num_workers, yolo_root=yolo_data_root,
                            yolo_exp_folds=yolo_exp_folds, treat_as_single_class=treat_as_single_class)
print(f"Cross validation")
print("-"*10)
crossvalidator = CNN_KCrossValidation(data_root=cnn_data_fold, k=k_tot, dataset = dataset, task=task)
crossvalidator._change_kfold(fold_i=k_i)

# processor
print("-"*10)
print(f"Data Loaders")
print("-"*10)
# if treat_as_single_class: 
#     map_classes = {'false_positives':0, 'item':1}
dataloader_cls = CNNDataLoaders(root_dir=cnn_data_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
dataloaders = dataloader_cls()

# raise NotImplementedError()

# Load the pretrained model from pytorch
print("-"*10)
print(f"VGG16 definition")
vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
vgg16.load_state_dict(torch.load(weights_path))
# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
vgg16.classifier[-1]= nn.Linear(num_features, len(map_classes))
vgg16.to(device)

# plot labels

# train model 
print("-"*10)
print(f"Train Model")
print("-"*10)
# model = vgg16
criterion = nn.BCEWithLogitsLoss()
optimizer_ft = torch.optim.SGD(vgg16.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
vgg16 = train_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, device=device, map_classes=map_classes,
                    criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epochs)
os.makedirs(weights_save_fold)
torch.save(vgg16.state_dict(), os.path.join(weights_save_fold, 'VGG16_v2-OCT_Retina_half_dataset.pt'))
eval_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, criterion=criterion)
