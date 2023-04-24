import time 
import torch
import os, shutil, sys
from glob import glob
from tqdm import tqdm
from cnn_feat_extract_loaders import CNNDataLoaders

def prepare_data(cnn_root_fold:str, map_classes:dict, batch:int, num_workers:int): 

    feat_extract_fold = os.path.join(cnn_root_fold, 'feat_extract')
    os.makedirs(feat_extract_fold, exist_ok=True)

    # get all images: 
    images = glob(os.path.join(cnn_root_fold, 'tiles', '*', '*', '*.jpg')) # get all images 
    class_fold_names = glob(os.path.join(cnn_root_fold, 'tiles', '*', '*/'))
    class_fold_names = set([os.path.split(os.path.dirname(fp))[1] for fp in class_fold_names])
    assert len(images)>0, f"No images like {os.path.join(cnn_root_fold, 'tiles', '*', '*', '*.jpg')}"

    # create folds:
    for fold in class_fold_names: 
        os.makedirs(os.path.join(feat_extract_fold, fold), exist_ok=True)

    # fill fold:
    for img in tqdm(images, desc="Filling 'feature_extract'"): 
        clss_fold_name = os.path.split(os.path.dirname(img))[1]
        dst = os.path.join(feat_extract_fold, clss_fold_name, os.path.basename(img))
        if not os.path.isfile(dst):
            shutil.copy(src=img, dst=dst)
    assert len(os.listdir(feat_extract_fold))>0, f"No images found in extract fold: {feat_extract_fold}"

    # map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
    # batch = 1
    # num_workers = 0
    dataloader_cls = CNNDataLoaders(root_dir=feat_extract_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
    dataloader = dataloader_cls()

    return  dataloader


def feature_extraction(model, dataloader, dataloader_cls, criterion):

    since = time.time()
    # feat_extract_fold = os.path.join(cnn_root_fold, 'feat_extract')
    # avg_loss = 0
    # avg_acc = 0
    # loss_test = 0
    # acc_test = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # tot_images = glob(os.path.join(feat_extract_fold, '*.jpg'))
    # test_batches = len(dataloaders['test'])
    print("Evaluating model")
    print('-' * 10)
    
    # for i, data in enumerate(dataloaders['test']):
    #     if i % 100 == 0:
    #         print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)


    for i, data in enumerate(dataloader):
            
        # if i % 100 == 0:
        #     print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        inputs, labels = data

        # inputs, labels = data
        inputs.to(device), labels.to(device)

        outputs = model(inputs)
        # print(f"\noutputs shape: {outputs.shape}")
        # print(f"labels shape: {labels.data.shape}")
        # print(f"labels shape: {labels.data.argmax(dim=1).shape}")
        _, preds = torch.max(outputs.data, 1, keepdim=True)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    # avg_loss = loss_test / dataloader_cls.dataset_size
    # avg_acc = acc_test / dataloader_cls.dataset_size
    
    # elapsed_time = time.time() - since
    # print()
    # print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    # print("Avg loss (test): {:.4f}".format(avg_loss))
    # print("Avg acc (test): {:.4f}".format(avg_acc))
    # print('-' * 10)

    return


if __name__ == "__main__": 

    cnn_root_fold = '/Users/marco/helical_tests/test_cnn_processor/test_crossvalidation'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
    batch = 1
    num_workers = 0
    dataloader = prepare_data(cnn_root_fold=cnn_root_fold, map_classes=map_classes, batch=batch, num_workers=num_workers)
    features = feature_extraction(model, dataloader, dataloader_cls, criterion)
