import plotly.io as pio
pio.renderers.default = "vscode"
import pandas as pd

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go

import os
from glob import glob


class TrainPlotter():

    def __init__(self, yolov5_root:str) -> None:
        assert os.path.isdir(yolov5_root), f"'train_exp_root':{yolov5_root} is not a valid dirpath."
        self.yolov5_root = yolov5_root
        
        return
    
    def _get_data(self) -> dict:
        """ Extracts csv tables from the subfolders and puts them into a pandas dataframe."""

        files = glob(os.path.join(self.yolov5_root, 'runs', 'train', '*', '*.csv'))
        data = [(os.path.basename(os.path.dirname(file)), file) for file in files]
        data = dict(data)

        return data
    
    def _load_dfs(self, data:dict):

        tables = []
        for exp_fn, exp_fp in data.items():
            if not os.path.isfile(exp_fp):
                print(f"❗️ exp_fp:{exp_fp} is not a valid filepath. Skipping.")
                continue
            df = pd.read_csv(exp_fp)
            df.columns = df.columns.str.strip()
            tables.append((exp_fn, df))
            # print(df.head())
            # raise NotImplementedError()
        tables = dict(tables)

        return tables
    
    def _plot_metrics(self, tables:dict) -> None:

        fig = go.Figure()
        for exp_fn, table in tables.items():
            # x = table['epochs']
            # y1 = table['metrics/mAP_0.5']
            # trace = px.add_trace(data_frame=table, x = 'epoch', y = 'metrics/mAP_0.5', title = exp_fn)
            fig.add_trace(go.Scatter(x = table['epoch'] , y = table['metrics/mAP_0.5'], mode ='lines', name = exp_fn))
        plot(fig)

        return
    
    def __call__(self) -> None:
        data = self._get_data()
        tables = self._load_dfs(data)
        self._plot_metrics(tables)
        return


def test_TrainPlotter():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    yolov5_root = '/Users/marco/yolov5' if system == 'mac' else r'C:\marco\yolov5'
    plotter = TrainPlotter(yolov5_root=yolov5_root)
    plotter()



    return


if __name__ == "__main__":
    test_TrainPlotter()

