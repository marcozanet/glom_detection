import torch 
import os
from torch import nn, optim
import time
from MIL_model import MIL_NN
from MIL_dataloader import get_loaders
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from MIL_utils import calculate_metric, print_scores
from utils import get_config_params
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


# #################    PARAMS    ####################
# PARAMS = get_config_params('mil_trainer')


##############    PREPROCESSING    ################
def preprocess(params:dict):
    print('*'*20)
    print("PREPARE DATA FOR MIL TRAINING")
    print('*'*20)
    root = params['root']
    all_slides_dir = params['all_slides_dir']
    map_classes = params['map_classes']
    bag_classes = params['bag_classes']
    bag_classes = {0:0.25, 1:0.5, 2:0.75, 3:1} # TODO ISSUE READING YAML
    n_instances_per_bag = params['n_instances_per_bag']
    stain = params['stain']
    batch = params['batch']
    limit_n_bags_to = params['limit_n_bags_to']
    num_workers = params['num_workers']
    train_loader_path = os.path.join(os.path.dirname(root),  'train_loader.pth')
    val_loader_path = os.path.join(os.path.dirname(root), 'val_loader.pth')
    feat_extract_folder_path = os.path.join(os.path.dirname(root), 'feat_extract')
    print(train_loader_path)
    print(val_loader_path)
    # load loaders if they exist:
    if os.path.isdir(feat_extract_folder_path):
        print(f"'feat_extract' folder existing.")
        if os.path.isfile(train_loader_path) and os.path.isfile(val_loader_path):
            print(f"Dataloaders loaded")
            train_loader = torch.load(train_loader_path)
            val_loader = torch.load(val_loader_path)
            return train_loader, val_loader

    # if they don't exist already, compute them:
    train_loader, val_loader =  get_loaders(root=root,
                                            all_slides_dir=all_slides_dir,
                                            map_classes=map_classes, 
                                            bag_classes=bag_classes, 
                                            n_instances_per_bag=n_instances_per_bag,
                                            stain=stain,
                                            batch = batch, 
                                            num_workers = num_workers,
                                            limit_n_bags_to=limit_n_bags_to)
    torch.save(train_loader, train_loader_path)
    torch.save(val_loader, val_loader_path)
    
    return train_loader, val_loader

#################    TRAIN    ####################
def train(params:dict, train_loader, val_loader):

    print('*'*20)
    print("START TRAINING")
    print('*'*20)
    epochs = params['epochs']
    # n_instances_per_bag = params['n_instances_per_bag']
    lr0 = params['lr0']
    batch_size = params['batch']
    ex_feats, ex_labels = next(iter(train_loader))
    assert ex_feats.shape[0] == batch_size, f"First shape of ex_feats = {ex_feats.shape[0]}, but batch shape is {batch_size}"
    # print(ex_feats.shape)
    feats_shape = ex_feats.shape[1:]
    # print(feats_shape)
    feats_shape = int(torch.prod(torch.tensor([el for el in feats_shape])))
    # model:
    model = MIL_NN(n=feats_shape, n_classes=4)
    # print(f"Instantiating model with n= {feats_shape}")
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean') # your loss function, cross entropy works well for multi-class problems
    # print('1')
    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9)
    losses = []
    batches = len(train_loader)
    print(f"Total batches train: {batches}")
    val_batches = len(val_loader)
    print(f"Total batches val: {val_batches}")

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        # print('appen dopo 4')
        for i, data in progress:

            X = data[0].to(device)
            # print(f"Batch shape: {X.shape}")
            y =  data[1].to(device)
            # print(f"trying to reshape to [{batch_size, 1, feats_shape}] = {batch_size*feats_shape}  ")
            X = X.reshape([batch_size, 1, feats_shape])
            # print(y)
            # print('6')
            # training step for single batch
            model.zero_grad() # to make sure that all the grads are 0 
            """
            model.zero_grad() and optimizer.zero_grad() are the same 
            IF all your model parameters are in that optimizer. 
            I found it is safer to call model.zero_grad() to make sure all grads are zero, 
            e.g. if you have two or more optimizers for one model.

            """
            outputs, _, _ = model(X) # forward
            loss = loss_function(outputs, y) # get loss
            loss.backward() # accumulates the gradient (by addition) for each parameter.
            optimizer.step() # performs a parameter update based on the current gradient 
           

            # TODO REMOVE calculate P/R/F1/A metrics for batch
            # prediced_classes = outputs.detach().argmax(dim=1)#.round()
            # print(prediced_classes)
            # print(y)
            # precision, recall, f1, accuracy = [], [], [], []
            # for acc, metric in zip((precision, recall, f1, accuracy), (precision_score, recall_score, f1_score, accuracy_score)):
            #     acc.append(calculate_metric(metric, y.cpu(), prediced_classes.cpu()))
            # print_scores(precision, recall, f1, accuracy, val_batches)
            # TODO REMOVE

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss
            # print('Done one')
            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                # print(X.shape)
                X = X.reshape([batch_size, 1, feats_shape])
                outputs, _, _ = model(X) # this get's the prediction from the network
                prediced_classes = outputs.detach().argmax(dim=1) #.round()
                # print(prediced_classes)
                # print(y)
                val_losses += loss_function(outputs, y)
                
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(calculate_metric(metric, y.cpu(), prediced_classes.cpu()))
            
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches) # for plotting learning curve
    print(f"Training time: {time.time()-start_ts}s")

def run():
    params = get_config_params('mil_trainer')
    train_loader, val_loader = preprocess(params)
    train(params=params, train_loader=train_loader, val_loader=val_loader)
    return



if __name__ == '__main__':
    run()