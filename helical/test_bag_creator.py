import os
from utils import get_config_params
from MIL_bags_creation import BagCreator

PARAMS = get_config_params('mil_trainer')
lr0 = PARAMS['lr0']
epochs = PARAMS['epochs']
train_img_dir = PARAMS['train_img_dir']
n_images_per_bag = PARAMS['n_images_per_bag']
n_classes = PARAMS['n_classes']
n_classes = PARAMS['n_classes']
batch = PARAMS['batch']
sclerosed_idx = PARAMS['sclerosed_idx']
num_workers = PARAMS['num_workers']
mapping = PARAMS['mapping']
val_img_dir = os.path.join(os.path.dirname(train_img_dir), 'val')
test_img_dir = os.path.join(os.path.dirname(train_img_dir), 'test')




creator = BagCreator(instances_folder=train_img_dir, 
                    sclerosed_idx=sclerosed_idx, 
                    exp_folder=exp_folder,
                    n_classes=n_classes,
                    n_images_per_bag=n_images_per_bag)
creator()

return