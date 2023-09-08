
import os 
import time
from typing import Any
from tqdm import tqdm
import torch
from datetime import datetime
from torch import nn
import seaborn as sns
from glob import glob
from sklearn.metrics import confusion_matrix
from utils import get_config_params
from cnn_trainer_base import CNN_Trainer_Base
import matplotlib.pyplot as plt



class CNN_Validator(CNN_Trainer_Base):

    def __init__(self, config_yaml_fp: str) -> None:
        super().__init__(config_yaml_fp)
        self.params = get_config_params(config_yaml_fp,'cnn_validator')
        self._set_new_attrs()
        os.makedirs(self.save_dir)
        return
    
    def _set_new_attrs(self)->None:
        self.model_path = self.params['model_path'] if self.params['model_path'] is not None else self._get_last_model_path()
        save_dir = self.params['save_dir']
        dt_string = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = os.path.join(save_dir, 'validate', f"{dt_string}")
        self.all_true_classes = []
        self.all_preds = []
        return
    
    
    def _get_last_model_path(self)->str:
        """ Gets last (modified on MacOS/created on Windows) fold."""
        func_n = self._get_last_model_path.__name__
        folds = [os.path.join(self.cnn_exp_fold, fold) for fold in os.listdir(self.cnn_exp_fold) if "DS" not in fold]
        last_exp = max(folds, key=os.path.getctime)
        assert os.path.isdir(last_exp), self.assert_log(f"CNN experiment last fold: {last_exp} is not a valid dirpath.", func_n=func_n)
        model_path = glob(os.path.join(last_exp, '*.pt'))
        assert len(folds)>0, self.assert_log(f"CNN experiment dir: {self.cnn_exp_fold} looks empty!", func_n=func_n)
        assert len(model_path)==1, self.assert_log(f"Found {len(model_path)} models within last CNN exp dir:{last_exp}", func_n=func_n)
        return model_path[0]    
    
    
    def plot_confusion_matrix(self, avg_loss:float, avg_acc:float)->None:
        """ Plots and saves confusion martix. """

        cm = confusion_matrix(y_true=self.all_preds, y_pred=self.all_true_classes) / len(self.all_true_classes)
        title = f"Avg Accuracy: {avg_acc:.2f}. Avg Loss {avg_loss:.2f}:"
        heatmap = sns.heatmap(data=cm, annot=True, fmt='.2f')
        fig = heatmap.get_figure()
        plt.title(title)
        fn = 'img_val_cm.png'
        fig.savefig(fn)
        fig.savefig(os.path.join(self.save_dir, fn))
        plt.close()
        return


    def eval_model(self)->None:
        func_n = self.eval_model.__name__

        # check there is at least 1 param requiring grad
        model = self.get_model()
        model.load_state_dict(torch.load(self.model_path))  # load trained model
        if self.device != 'cpu': model.to(self.device)   # move model to gpus
        assert any([param.requires_grad for param in model.features.parameters()]), self.assert_log(f"No param requires grad. Model won't update.", func_n=func_n)
        criterion = nn.CrossEntropyLoss()
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_val = 0
        acc_val = 0

        # test_batches = len(self.dataloaders['val'])
        print("Evaluating model")
        print('-' * 10)
        get_desc = lambda acc_val_i, loss_val_i: f"Acc val: {acc_val_i:.2f}, Loss val: {loss_val_i:.2f}."
        progress_bar = tqdm(self.loaders['val'], desc=get_desc(acc_val_i=0, loss_val_i=0))
        for i, data in enumerate(progress_bar):

            model.train(False)
            model.eval()
            inputs, labels = data
            v_batch = labels.shape[0]

            # move data to gpu
            if self.device != 'cpu': inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1, keepdim=True)         # get pred from logits
            _, true_classes = torch.max(labels.data, 1, keepdim=True)   # get true class 

            self.all_preds.extend(list(preds.cpu().detach().numpy()))
            self.all_true_classes.extend(list(true_classes.cpu().detach().numpy()))

            if i%25==0: 
                self.show_train_data(images=inputs, pred_lbl=preds, gt_lbl=true_classes, mode='val')
            loss_val += loss.data
            acc_val_i = torch.sum(preds == true_classes).cpu().numpy()/v_batch
            acc_val += acc_val_i
            if i%1==0: progress_bar.set_description(desc=get_desc(acc_val_i=acc_val_i, loss_val_i=loss.data))
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        # plot
        avg_loss = loss_val / (i+1) #dataloader_cls.valset_size
        avg_acc = acc_val / (i+1) #dataloader_cls.valset_size
        self.plot_confusion_matrix(avg_loss=avg_loss, avg_acc=avg_acc)


        # print duration
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)

        return
    
    def __call__(self) -> Any:
        self.get_loaders(mode='val')
        self._get_last_model_path()
        self.eval_model()

        return 