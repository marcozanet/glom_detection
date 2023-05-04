import os
from utils import get_config_params
from MIL_bags_creation_new import BagCreator

PARAMS = get_config_params('mil_trainer')
lr0 = PARAMS['lr0']
epochs = PARAMS['epochs']
instances_root = PARAMS['instances_root']
# train_instances_dir = os.path.join(instances_root)
n_instances_per_bag = PARAMS['n_instances_per_bag']
map_classes = PARAMS['map_classes']
n_classes = PARAMS['n_classes']
n_classes = PARAMS['n_classes']
batch = PARAMS['batch']
sclerosed_idx = PARAMS['sclerosed_idx']
num_workers = PARAMS['num_workers']
mapping = PARAMS['mapping']
# val_img_dir = os.path.join(os.path.dirname(train_instances_dir), 'val')
# test_img_dir = os.path.join(os.path.dirname(train_instances_dir), 'test')




creator = BagCreator(instances_folder=instances_root, 
                     sclerosed_idx=sclerosed_idx, 
                     map_classes=map_classes,
                    #  exp_folder=exp_folder,
                     n_classes=n_classes,
                     n_instances_per_bag=n_instances_per_bag)
creator()

# return