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


class Plotter(Profiler): 

    def __init__(self,
                files:list = None,
                *args, 
                **kwargs) -> None:
        """ Data Plotter to help visualize a data overview. 
            Super class Profiler needs a root folder structured like: root -> wsi/tiles->train,val,test->images/labels."""

        super().__init__(*args, **kwargs)
        self.df = self._get_df()
        if files is not None:
            assert all([os.path.isfile(file) for file in files])
            assert all(['png' in file for file in files])
        self.files = files

        return
    
    def _plot_instance_sizes(self):
        
        df = self.df
        df['width'] = df['width']*100
        df['width'] = df['width'].astype(int)
        df['height'] = df['height']*100
        df['height'] = df['height'].astype(int)               
        
        sns.lmplot(data=self.df, x='width', y = 'height',hue = 'class_n', 
                height=10, palette=['green','red'], legend=True)
        return
    
    def _show_random_sets(self):

        for set in ['train', 'val', 'test']:
            self._show_random_tiles(set = set)
        
        return
    

    def _show_random_tiles(self, set = 'train'):
        """ Shows 3 random images/labels. """

        if self.files is None: 
            labels = self.data['tile_labels']
            labels_fold = os.path.join(self.data_root, set, 'labels')
            rand_idx_1 = np.random.randint(0, len(labels))
            rand_idx_2 = np.random.randint(0, len(labels))
            rand_idx_3 = np.random.randint(0, len(labels))
            labels = os.path.join(labels_fold, labels[rand_idx_1]), os.path.join(labels_fold, labels[rand_idx_2]), os.path.join(labels_fold, labels[rand_idx_3])
            images = labels[0].replace('labels', 'images').replace('.txt', '.png'), labels[1].replace('labels', 'images').replace('.txt', '.png'), labels[2].replace('labels', 'images').replace('.txt', '.png')
        else:

            images = self.files
            labels = [file.replace('images', 'labels').replace('.png', '.txt') for file in images]

        # pick random labels/images
        # labels_fold = os.path.join(self.save_root, 'labels')
        # labels = [os.path.join(file) for file in os.listdir(labels_fold)]

        # show image + rectangles on labels:
        plt.figure(figsize=(20, 60))
        for i, (image_fp, label_fp) in enumerate(zip(images, labels)):

            image = cv2.imread(image_fp)
            W, H = image.shape[:2]

            # read label
            with open(label_fp, 'r') as f:
                text = f.readlines()
                f.close()
            
            # draw rectangle for each glom/row:
            for row in text: 
                items = row.split(sep = ' ')
                class_n = int(items[0])
                xc, yc, box_w, box_h = [float(num) for num in items[1:]]
                w_true, h_true = box_w, box_h
                xc, box_w = xc * W, box_w * W
                yc, box_h = yc * H, box_h * H
                x0, x1 = int(xc - box_w // 2), int(xc + box_w // 2)
                y0, y1 = int(yc - box_h//2), int(yc + box_h//2)
                start_point = (x0, y0)
                end_point = (x1,y1)
                color = (0,255,0) if class_n == 0 else (255,0,0)
                text = f'{round(w_true,2), round(h_true,2)}' if class_n == 0 else f'{round(w_true,2), round(h_true,2)}' # healthy, unhealthy
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.rectangle(img = image, pt1 = start_point, pt2 = end_point, color = color, thickness=2)
                image = cv2.putText(image, text, org = (x0,y0-H//50), color=color, thickness=2, fontFace=font, fontScale=1)

            # add subplot with image
            plt.subplot(1,3,i+1)
            plt.title(f"Example tiles - {set} set")
            plt.imshow(image, interpolation='nearest')
            plt.axis('off')
        
        plt.show()

        return
    
    def _plot_area(self): 

        sns.displot(self.df, col='fold', x="area")     


        return

    def _plot_height_width(self): 

        sns.displot(self.df, col='fold',x="width")  
        sns.displot(self.df, col='fold',x="height")  


        return
    
    def _plot_fullempty(self):

        sns.catplot(data=self.df, kind="bar",
                    x="fold", y="area", hue="class_n",
                    errorbar="sd", palette="dark", alpha=.6, height=6)   


        return

    
    def _plot_gloms_per_sample(self):

        df = pd.DataFrame(data = self.gloms_slides, columns = ['sample', 'n_gloms'])
        fig = sns.barplot(df, x = df.index, y = 'n_gloms', color = 'black')
        plt.title('Barplot #gloms per tissue sample')
        plt.xlabel('#gloms_per_sample')
        plt.show()

        return


    
    def __call__(self) -> None:

        self._plot_area()
        self._plot_height_width()

        self._show_random_sets()

        # self._plot_instance_sizes()

        # self._plot_area()

        # self.show_summary()

        return



def test_Plotter():
    system = 'mac'
    data_root = '/Users/marco/Downloads/train_20feb23' if system == 'mac' else r'D:\marco\datasets\muw\detection'
    files = ['/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_31_18.png',
            '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_52_12.png',
            '/Users/marco/Downloads/train_20feb23/tiles/train/images/200209761_09_SFOG_sample0_49_24.png']
    profiler = Plotter(data_root=data_root, files=None)
    profiler()

    return


if __name__ == '__main__':
    test_Plotter()