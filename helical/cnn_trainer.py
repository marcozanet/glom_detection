import torch, torchvision
from torchvision import models
from torch import nn
from cnn_train_func import train_model, eval_model
from cnn_loaders import CNNDataLoaders
import os
from crossvalidation import KCrossValidation

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# params
device='cuda:0' if torch.cuda.is_available() else 'cpu'
root_dir = '/Users/marco/helical_tests/test_cnn_processor/test_processor/tiles'
map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
k_tot = 4
k_i = 1
batch = 2
epochs = 10
num_workers = 0
weights_path = '/Users/marco/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth'
dataset = 'muw'
task = 'detection'

# processor
dataloader_cls = CNNDataLoaders(root_dir=root_dir, map_classes=map_classes, batch=batch, num_workers=num_workers)
dataloaders = dataloader_cls()
crossvalidator = KCrossValidation(data_root=os.path.dirname(root_dir), k=k_tot, dataset = dataset, task=task)
crossvalidator._change_kfold(fold_i=k_i)

# Load the pretrained model from pytorch
vgg16 = models.vgg16_bn(weights = 'VGG16_BN_Weights.DEFAULT')
vgg16.load_state_dict(torch.load(weights_path))
print(vgg16)

raise NotImplementedError()
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
optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
vgg16 = train_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, 
                    criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epochs)
torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')
eval_model(model=vgg16, dataloader_cls=dataloader_cls, dataloaders=dataloaders, criterion=criterion)
