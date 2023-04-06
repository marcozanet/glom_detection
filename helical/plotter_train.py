import plotly.io as pio
pio.renderers.default = "vscode"
import pandas as pd

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import random
import os
from glob import glob

COLORS = [      'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
                'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
                'blueviolet', 'brown', 'burlywood', 'cadetblue',
                'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
                'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
                'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
                'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                'darkslateblue', 'darkslategray', 'darkslategrey',
                'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
                'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
                'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
                'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
                'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
                'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
                'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
                'lightgoldenrodyellow', 'lightgray', 'lightgrey',
                'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
                'lightskyblue', 'lightslategray', 'lightslategrey',
                'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
                'linen', 'magenta', 'maroon', 'mediumaquamarine',
                'mediumblue', 'mediumorchid', 'mediumpurple',
                'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
                'mediumturquoise', 'mediumvioletred', 'midnightblue',
                'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
                'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
                'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise']


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
            print(df.head())
            # raise NotImplementedError()
        tables = dict(tables)

        return tables
    
    def _plot_metrics(self, tables:dict) -> None:

        fig = go.Figure()
        n_colors = len(tables.values())
        colors = random.sample(COLORS, k=n_colors)

        for i, (exp_fn, table) in enumerate(tables.items()):
            # x = table['epochs']
            # y1 = table['metrics/mAP_0.5']
            # trace = px.add_trace(data_frame=table, x = 'epoch', y = 'metrics/mAP_0.5', title = exp_fn)
            fig.add_trace(go.Scatter(x = table['epoch'] , y = table['metrics/precision'], line = dict(width=4, color=colors[i]), name = exp_fn))
            fig.add_trace(go.Scatter(x = table['epoch'] , y = table['metrics/recall'], line = dict(width=4, dash='dot', color=colors[i]), name = exp_fn))
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
    
    yolov5_root = '/Users/marco/Downloads/exps_up_to_27.03.23' if system == 'mac' else r'C:\marco\yolov5'
    plotter = TrainPlotter(yolov5_root=yolov5_root)
    plotter()



    return


if __name__ == "__main__":
    test_TrainPlotter()

