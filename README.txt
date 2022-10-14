#####################
1. INSTALLATION
#####################
- create new env 
- install pytorch and pytorchvision 
- install yolo and all requirements 
- install segmentation models pytorch  
- install pytorch-lightning 
- install skimage 
= install patchify 
- install openslide NB install with conda on MacOS so that installs both 
- install PyYAML



Pipeline Runner.
1) Data preparation: from WSI and their annotations in JSON format to patches and relative annotations in txt format (suitable for YOLO)
2) Manually run YOLO 
3) Stitch back patches to get WSI prediction
4) Crop images around prediction from YOLO
5) Feed cropped images to U-Net 

1) data preparation for yolo: from slides/slides annotation to patches/patches annotations in yolo format (txt). OK
    script: wsi_segm_gejson_to_patch_segm_png.py
2) data preparation for unet: from slides/slides annotation to patches/patches annotations in UNet format (png). OK
    script: get_imgs_masks_tiles.py
3) application of yolo --> predictions in txt format --> using a FIXED shape, crop output predictions (to be fed to unet). OK
    cd yolov5
    conda activate torch-
    python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
4) crop patch annotations (segmentations) in the same way to have annotations for unet 



Parto con:
wsi e le loro annotations divise in train e test (NB sia wsi che annotation nella stessa folder)
1) convert slides and annotations to suitable format for YOLO, then patchify both: wsi_segm_gejson_to_tile_bb_txt.py
2) 
