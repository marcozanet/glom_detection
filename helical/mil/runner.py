import torch 
from torch import nn, optim
import inspect
import time
from MIL_model import MIL_NN
from MIL_dataloader import get_loaders
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score




def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
# 4. Train & Test
# In [60]:
import numpy as np
start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

lr0 = 1e-4

# model:
model = MIL_NN().to(device)

# params you need to specify:
epochs = 10

# get loaders:
train_img_dir = '/Users/marco/helical_tests/test_bagcreator/images'
val_img_dir = train_img_dir
test_img_dir = train_img_dir
train_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
val_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
test_detect_dir = '/Users/marco/yolov5/runs/detect/exp7'
n_images_per_bag = 9
n_classes = 4
batch = 1
sclerosed_idx = 2
num_workers = 0
mapping = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}

train_loader, val_loader = get_loaders(train_img_dir=train_img_dir,
                                        train_detect_dir=train_detect_dir, 
                                        # val_img_dir=val_img_dir,
                                        # val_detect_dir=val_detect_dir,
                                        n_images_per_bag=n_images_per_bag,
                                        n_classes=n_classes,
                                        test_img_dir=test_img_dir,
                                        test_detect_dir=test_detect_dir,
                                        sclerosed_idx=sclerosed_idx,
                                        batch=batch,
                                        num_workers=num_workers,
                                        mapping=mapping)
loss_function = torch.nn.BCELoss(reduction='mean') # your loss function, cross entropy works well for multi-class problems


#optimizer = optim.Adadelta(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9)

losses = []
batches = len(train_loader)
val_batches = len(val_loader)

# loop for every epoch (training + evaluation)
for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  -------------------- 
    # set model to training
    model.train()
    for i, data in progress:
        # print(type(data))
        # print(type(data[0]))
        # print(data[0][0])
        X = data[0].to(device)
        y =  data[1].to(device)
        X = X.reshape([1,20*20])
        y = y.type(torch.FloatTensor)
        # training step for single batch
        model.zero_grad() # to make sure that all the grads are 0 
        """
        model.zero_grad() and optimizer.zero_grad() are the same 
        IF all your model parameters are in that optimizer. 
        I found it is safer to call model.zero_grad() to make sure all grads are zero, 
        e.g. if you have two or more optimizers for one model.

        """
        outputs = model(X) # forward
        loss = loss_function(outputs, y) # get loss
        loss.backward() # accumulates the gradient (by addition) for each parameter.
        optimizer.step() # performs a parameter update based on the current gradient 

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

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
            print('sending to ghe moon')
            X = X.reshape([1,7*512])
            y = y.type(torch.cuda.FloatTensor)
            outputs = model(X) # this get's the prediction from the network
            prediced_classes =outputs.detach().round()
            #y_pred.extend(prediced_classes.tolist())
            val_losses += loss_function(outputs, y)
            
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy), 
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), prediced_classes.cpu())
                )
          
    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss/batches) # for plotting learning curve
print(f"Training time: {time.time()-start_ts}s")