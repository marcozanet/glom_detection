import os, shutil
from glob import glob
from cnn_dataset import CNNDataset
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt 
import torchvision

class CNNDataLoaders():

    def __init__(self,
                 root_dir:str,
                 map_classes: dict,
                 batch:int = 2,
                 num_workers:int = 0,
                 ) -> None:
        
        self.root_dir = root_dir
        self.map_classes = map_classes
        self.batch = batch 
        self.num_workers = num_workers
        self._parse()

        return
    

    def _parse(self): 

        assert os.path.isdir(self.root_dir), f"'root_dir':{self.root_dir} is not a valid dirpath."
        assert isinstance(self.batch, int), f"'batch':{self.batch} should be int."
        assert isinstance(self.num_workers, int), f"'num_workers':{self.num_workers} should be int."
        assert isinstance(self.map_classes, dict), f"'map_classes':{self.map_classes} should be dict."

        return
    
    def get_loaders(self):

        # get train, val, test set:
        trainset = CNNDataset(root_dir=self.root_dir, dataset='train', map_classes=self.map_classes) 
        valset = CNNDataset(root_dir=self.root_dir, dataset='val', map_classes=self.map_classes)
        testset = CNNDataset(root_dir=self.root_dir, dataset='test', map_classes=self.map_classes)

        # check that datasets don't intersect:
        if valset.imgs_fn is not None:
            assert set(trainset.imgs_fn).isdisjoint(set(valset.imgs_fn)) 
            # assert set(valset.imgs_fn).isdisjoint(set(testset.imgs_fn))
        # assert set(trainset.imgs_fn).isdisjoint(set(testset.imgs_fn))

        self.trainset_size = len(trainset)
        self.valset_size = len(valset)
        self.testset_size = len(testset)

        print(f"Train size: {len(trainset)} images.")
        print(f"Valid size: {len(valset)} images." )
        print(f"Test size: {len(testset)} images.")

        train_dataloader = DataLoader(trainset, batch_size=self.batch, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        valid_dataloader = DataLoader(valset, batch_size=self.batch, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        # print(type(valid_dataloader))
        test_dataloader = DataLoader(testset, batch_size=self.batch, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        dataloaders = {'train':train_dataloader, 'val':valid_dataloader, 'test':test_dataloader} if valset.imgs_fn is not None else {'train':train_dataloader, 'test':test_dataloader}
        self.dataloaders = dataloaders
        # print(dataloaders)

        return dataloaders
    
    
    def show_data(self):

        def imshow(inp, title=None):
            inp = inp.numpy().transpose((1, 2, 0))

            # plt.figure(figsize=(10, 10))
            fig = plt.figure()
            plt.axis('off')
            # plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)
            fig.savefig('cnn_crops.png')

        def show_databatch(inputs, classes):
            out = torchvision.utils.make_grid(inputs)
            
            onehot2int = lambda tensor: tensor.argmax() 

            reversed_map_classes = {v:k for k,v in self.map_classes.items()}

            # classes
            imshow(out, title=[reversed_map_classes[int(onehot2int(x))] for x in classes])

        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))
        # classes = [self.map_classes[self.] for x in classes]
        # print(classes)
        show_databatch(inputs, classes)


        return 
    
    def __call__(self) -> tuple:  

        dataloaders = self.get_loaders()
        self.show_data()
        self.show_data()
        self.show_data()

        return dataloaders


if __name__ == "__main__": 
    root_dir = '/Users/marco/helical_tests/test_cnn_processor'
    map_classes = {'Glo-healthy':0, 'Glo-unhealthy':1, 'false_positives':2} 
    batch = 3
    num_workers = 0
    dataloader = CNNDataLoaders(root_dir=root_dir, map_classes=map_classes, batch=batch, num_workers=num_workers)
    dataloaders = dataloader()


    

# def get_loaders(root_dir:str, 
#                 batch = 2, num_workers = 8, resize = False, 
#                 classes = 2, mapping: dict = {(0, 0, 0): 0, (0, 255, 0): 1, (255, 0, 0): 2}) -> Tuple:

#     assert os.path.isdir(root_dir), f"'root_dir':{root_dir} is not a valid dirpath."
#     assert isinstance(batch, int), f"'batch':{batch} should be int."
#     assert isinstance(num_workers, int), f"'num_workers':{num_workers} should be int."
#     assert isinstance(resize, bool), f"'resize':{resize} should be boolean."
#     assert isinstance(classes, int), f"'classes':{classes} should be int."
#     assert isinstance(mapping, dict), f"'mapping':{mapping} should be dict."

#     # get train, val, test set:
#     trainset = CNNDataset(img_dir=root_dir, dataset='train')
#     valset = CNNDataset(img_dir=root_dir, dataset='val')
#     testset = CNNDataset(img_dir=root_dir, dataset='test')

#     # check that datasets don't intersect:
#     assert set(trainset.imgs_fn).isdisjoint(set(valset.imgs_fn))
#     assert set(valset.imgs_fn).isdisjoint(set(testset.imgs_fn))
#     assert set(trainset.imgs_fn).isdisjoint(set(testset.imgs_fn))

#     print(f"Train size: {len(trainset)} images.")
#     print(f"Valid size: {len(valset)} images." )
#     print(f"Test size: {len(testset)} images.")

#     train_dataloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
#     valid_dataloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
#     test_dataloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

#     data = next(iter(train_dataloader))
#     image = data['image']
#     print(f"image shape: {image.shape}")
    

#     return train_dataloader, valid_dataloader, test_dataloader