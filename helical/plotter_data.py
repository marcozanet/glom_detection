import os 
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np
import pandas as pd 
from tqdm import tqdm
from profiler import Profiler
import cv2
from loggers import get_logger
from decorators import log_start_finish
import random


class Plotter(Profiler): 

    def __init__(self,
                files:list = None,
                *args, 
                **kwargs) -> None:
        """ Data Plotter to help visualize a data overview. 
            Super class Profiler needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels."""

        super().__init__(*args, **kwargs)
        self.df_instances = self._get_instances_df()
        self.log.info(self.df_instances.head())
        self.df_tiles = self._get_tiles_df()
        self.log.info(self.df_tiles.head())

        if files is not None:
            assert all([os.path.isfile(file) for file in files])
            assert all(['png' in file for file in files])
        self.files = files

        return
    
    def _plot_instance_sizes(self) -> None:

        # print("########################################################")
        # print(self.__class__.__name__)
        
        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_instance_sizes', msg = f"Plotting sizes" )
        def do():

            df = self.df_instances
            df['width'] = df['width']*100
            df['width'] = df['width'].astype(int)
            df['height'] = df['height']*100
            df['height'] = df['height'].astype(int)               
            
            sns.lmplot(data=self.df_instances, x='width', y = 'height',hue = 'class_n', 
                    height=10, palette=['green','red'], legend=True)

            return
        
        do()

        return
    
    def _show_random_sets(self) -> None:

        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_instance_sizes', msg = f"Plotting random sets" )
        def do():       

            for set in ['train', 'val', 'test']:
                self._show_random_tiles(set = set)
            
        do()        

        return
    

    def _show_random_tiles(self, set = 'train', k:int = 6) -> None:
        """ Shows 3 random images/labels in subplots. """

        k = 3
        # @log_start_finish(class_name=self.__class__.__name__, func_name='_show_random_tiles', msg = f"Plotting random tiles" )
        def do(k:int):   
            
            if k%2 != 0:
                self.log.error(f"{self._class_name}.{'_show_random_tiles'}: K should be divisible by 2. Using default K=6 instead")
                k = 6

            replace_dir = lambda fp, to_dir, format: os.path.join(os.path.dirname(os.path.dirname(fp)), to_dir, os.path.basename(fp).split('.')[0] + f".{format}")

            # 1) Picking images:
            if self.files is None: # i.e. random images
                labels = self.data['tile_labels']
                labels = random.sample(labels, k=k)
                pairs = [(replace_dir(fp, to_dir='images', format='png'), fp) for fp in labels]
                pairs = list(filter(lambda pair: (os.path.isfile(pair[0]) and os.path.isfile(pair[1])), pairs))
            else: # i.e. specified images
                images = self.files
                pairs = [(replace_dir(fp, to_dir='labels', format='txt'), fp) for fp in images]
                pairs = list(filter(lambda pair: (os.path.isfile(pair[0]) and os.path.isfile(pair[1])), pairs))
            
            # 2) Show image/drawing rectangles as annotations:
            plt.figure(figsize=(20, k//2*10))
            for i, (image_fp, label_fp) in enumerate(pairs):

                self.log.info(f"i:{i}")

                # read image
                image = cv2.imread(image_fp)
                W, H = image.shape[:2]
                # read label
                with open(label_fp, 'r') as f:
                    text = f.readlines()
                    f.close()
                self.log.info("successfully read label")
                # draw rectangle for each glom/row:
                for row in text: 
                    items = row.split(sep = ' ')
                    class_n = int(float(items[0]))
                    xc, yc, box_w, box_h = [float(num) for num in items[1:]]
                    xc, box_w = xc * W, box_w * W
                    yc, box_h = yc * H, box_h * H
                    x0, x1 = int(xc - box_w // 2), int(xc + box_w // 2)
                    y0, y1 = int(yc - box_h//2), int(yc + box_h//2)
                    start_point = (x0, y0)
                    end_point = (x1,y1)
                    color = (255,0,0) if class_n == 0 else (0,255,0) 
                    text = 'unhealthy' if class_n == 0 else 'healthy'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.rectangle(img = image, pt1 = start_point, pt2 = end_point, color = color, thickness=2)
                    self.log.info("done rectangle")
                    image = cv2.putText(image, text, org = (x0,y0-H//50), color=color, thickness=2, fontFace=font, fontScale=1)
                    self.log.info("successfully done text")

                # add subplot with image

                self.log.info('adding subplot')
                self.log.info(f"k//2: {k//2}, k:{k}" )
                plt.subplot(k//2,2,i+1)
                self.log.info('adding title')
                plt.title(f"Example tiles - {set} set")
                self.log.info('plotting image with interpolation nearest')
                plt.imshow(image)
                self.log.info('axis off')
                plt.tight_layout()
                plt.axis('off')
            
            plt.show()
        
            return

        do(k=k)

        return
    

    def _plot_area(self) -> None: 
        """ Plots area with a distplot. """

        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_instance_sizes', msg = f"Plotting area" )
        def do():   
            
            # Plot:
            sns.displot(self.df_instances, col='fold', x="area")     
        
            return
        do()

        return


    def _plot_height_width(self) -> None: 
        """ Plots height and width distribution on a distplot."""

        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_height_width', msg = f"Plotting w,h" )
        def do():  
            # Plot width
            sns.displot(self.df_instances, col='fold',x="width") 
            # Plot height 
            sns.displot(self.df_instances, col='fold',x="height")  

            return

        do()

        return
    
    def _plot_fullempty(self) -> None:
        """ Plots emtpy, healthy_glom or unhealthy_glom distribution on a histplot."""
        print("########################################################")
        print(self.__class__.__name__)
        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_height_width', msg = f"Plotting w,h" )
        def do():  
            df = self.df_tiles
            print(df['class_n'].dtype)
            print(df['fold'])
            sns.histplot(data=df, x="fold", hue="class_n", multiple="dodge", shrink=.8)

            return
        
        do()

        return

    
    def _plot_gloms_per_sample(self) -> None:
        """ Plots gloms per sample in a barplot. """

        @log_start_finish(class_name=self.__class__.__name__, func_name='_plot_height_width', msg = f"Plotting w,h" )
        def do():  

            df = pd.DataFrame(data = self.gloms_slides, columns = ['sample', 'n_gloms'])
            sns.barplot(df, x = df.index, y = 'n_gloms', color = 'black')
            plt.title('Barplot #gloms per tissue sample')
            plt.xlabel('#gloms_per_sample')
            plt.show()

            return
        
        do()

        return


    
    def __call__(self) -> None:

        self._show_random_tiles()

        # self._plot_area()
        # self._plot_height_width()

        # self._show_random_sets()

        # self._plot_instance_sizes()

        # self._plot_area()

        # self.show_summary()

        return



def test_Plotter():
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'
    
    data_root = '/Users/marco/helical_tests/test_process_data_and_train/test_3_slides/detection' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    # files = ['/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_31_18.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_52_12.png',
    #         '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_49_24.png']
    plotter = Plotter(data_root=data_root, files=None, verbose = False)
    plotter()

    return


if __name__ == '__main__':
    test_Plotter()