import os 
import glob 
from typing import Literal
import shutil
from tqdm import tqdm


class Selector(): 

    def __init__(self,
                root:str, 
                save_folder:str,
                slide_format:Literal['.tif_pyr.tif'], 
                label_format:Literal['.mrxs.gson'], 
                stain = "SFOG", 
                copy:bool = True,
                clear_others:bool = False, 
                verbose:bool=False) -> None: 

        """ Pairs up files from MUW -> selects what files have a correspondent annotation file, 
            with option of deleting the other ones. """

        ALLOWED_SLIDE_FORMATS =  '.tif_pyr.tif'
        ALLOWED_LABEL_FORMATS =  '.mrxs.gson'

        assert os.path.isdir(root), f"'root' is not a valid dirpath."
        assert os.path.isdir(save_folder), f"'save_folder' is not a valid dirpath."
        assert slide_format in ALLOWED_SLIDE_FORMATS, f"'slide_format' should be one of {ALLOWED_SLIDE_FORMATS}. "
        assert label_format in ALLOWED_LABEL_FORMATS, f"'label_format' should be one of {ALLOWED_LABEL_FORMATS}. "
        assert isinstance(copy, bool), f"'safe_copy':{copy}."
        assert isinstance(verbose, bool), f"'verbose':{verbose}."

        self.root = root 
        self.label_format = label_format
        self.slide_format = slide_format
        self.copy = copy
        self.stain = stain
        self.verbose = verbose
        self.clear_others = clear_others
        self.save_folder = save_folder
        self.slides, self.labels = self._get_slides_labels()

        return


    def _get_slides_labels(self) -> list:

        # get all labels:
        labels = glob.glob(os.path.join(self.root, f'*{self.label_format}'))
        labels = [label for label in labels if os.stat(label).st_size > 1000 and self.stain in label ]
        labels = list(set(labels))

        # get all slides:
        slides = []
        for file in glob.glob(os.path.join(self.root, f'**/*{self.slide_format}'), recursive=True):
            if self.stain in file:
                slides.append(file)
        
        assert len(slides) > 0, f"0 slides found in {self.root}."
        assert len(labels) > 0, f"0 labels found in {self.root}."

        if self.verbose is True: 
            print(f"Found {len(slides)} pyramidal slides and {len(labels)} non empty labels.")

        return slides, labels


    def _get_matches(self):

        # get potential slide matches for existing labels:
        pot_pairs = [(label.replace('.mrxs.gson','_Wholeslide_Default_Extended.tif_pyr.tif'), label) for label in self.labels]
        pot_pairs = [(os.path.join(self.root, os.path.basename(label.split('.')[0]), os.path.basename(slide)), label) for slide, label in pot_pairs]
        matches = []
        missing_slides = []
        for slide, label in pot_pairs:
            if os.path.isfile(slide) and os.path.isfile(label):
                matches.append((slide, label))
            elif (not os.path.isfile(slide)) and os.path.isfile(label):
                missing_slides.append(slide)
        
        print(f"Matched pairs: {len(matches)}")
        print(f"Found annotations, but not pyramidal slides (maybe only not converted slide exists?) for {len(missing_slides)} pairs.")

        # print(f"Missing pyramidal slides: {missing_slides}")

        return matches



    def _move_pairs(self, matches: list):

        # copy/move matched slides,labels to save_folder:
        for slide, label in tqdm(matches):
            dst_label = os.path.join(self.save_folder, os.path.basename(label)).replace('.mrxs', '')
            dst_slide = os.path.join(self.save_folder, os.path.basename(slide)).replace('_Wholeslide_Default_Extended', '').replace('.tif_pyr', '')
            if not os.path.isfile(dst_label):
                if self.copy is True:
                    shutil.copy(src=label, dst=dst_label)
                else:
                    shutil.move(src=label, dst=dst_label)
            if not os.path.isfile(dst_slide):
                if self.copy is True:
                    shutil.copy(src=slide, dst=dst_slide)
                else:
                    shutil.move(src=slide, dst=dst_slide)

        return
    
    def __call__(self):
        
        # 1) get matching pairs:
        matches = self._get_matches()
        # 2) move pairs to new location:
        self._move_pairs(matches=matches)

        return


def test_Selector():

    print(" ########################    TEST __init__: ⏳    ########################")
    root = '/Users/marco/converted_30_01_23'
    slide_format = '.tif_pyr.tif'
    label_format = '.mrxs.gson'
    save_folder = '/Users/marco/Downloads/muw'
    selector= Selector(root=root,
                       save_folder=save_folder,
                       slide_format=slide_format,
                       label_format=label_format,
                       copy=True,
                       clear_others=True,
                       verbose=True)
    print(" ########################    TEST __init__: ✅    ########################")
    # print(" ########################    TEST _get_slides_labels: ⏳    ########################")
    # slides, labels = selector._get_slides_labels()
    # print(" ########################    TEST _get_slides_labels: ✅    ########################")
    print(" ########################    TEST _get_matches: ⏳    ########################")
    matches = selector._get_matches()
    print(" ########################    TEST _get_matches: ✅    ########################")
    print(" ########################    TEST _move_pairs: ⏳    ########################")
    selector._move_pairs(matches=matches)
    print(" ########################    TEST _move_pairs: ✅    ########################")
    return

if __name__ == '__main__':
    test_Selector()