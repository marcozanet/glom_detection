import os 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

from configurator import Configurator
from utils import get_config_params


class PrognosisReporter(Configurator):
    def __init__(self, config_yaml_fp:str) -> None:
        super().__init__()
        self.params = get_config_params(config_yaml_fp, 'prognosis_reporter')
        self._set_attr()
        self._parse_args()
        return
    
    def _set_attr(self)->None:
        for key, value in self.params.items():
            setattr(self, key, value )
        return
    
    def _parse_args(self)->None:
        func_n = self._parse_args.__name__
        assert os.path.isfile(self.data_csv), self.assert_log(f"'data_csv':{self.data_csv} is not a valid filepath.", func_n=func_n)
        return
    
    def prepare_data(self):
        df = pd.read_csv(self.data_csv, index_col=0)
        y = df['label'].values
        X = df.drop('label', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True, stratify=y)
        print(f"Patients in train set: {len(x_train)}. Patients in test se: {len(x_test)}")
        model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5)
        model.fit(X=x_train, y=y_train)
        y_pred = model.predict(X=x_test)
        t_acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Accuracy on test set: {t_acc:.2f}")
        plt.figure(figsize=(20, 10))
        plot_tree(model, max_depth=2, feature_names=X.columns)
        plt.savefig('img_decisiontree.png')

        return
    
    def __call__(self)->None:

        self.prepare_data()

        return
    



if __name__== '__main__':
    config_yaml_fp = '/Users/marco/yolo/code/helical/config_tcd.yaml'
    reporter = PrognosisReporter(config_yaml_fp=config_yaml_fp)
    reporter()