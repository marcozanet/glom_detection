import os
import numpy as np
from skimage import io
from tqdm import tqdm


def find_box_max_from_file(file):
    ''' Find max boxes dimensions from txt annotation. '''

    with open(file, 'r') as f:
        text = f.read()
        text = text.replace('\n', ' ')
        text = text.split(sep = ' ')
        if text[-1] == ' ':
            text = text[:-1]

        # remove '' at the end if not divisible by 0:
        if len(text) % 5 != 0:
            del_idx = [i for i in range(len(text)) if text[i] == '']
            del_idx = del_idx[0]
            text.pop(del_idx)
        assert len(text) % 5 == 0, f'Text has object {len(text)}, not divisible by 5.'

        nums = [float(num) for num in text]
        n_rows = len(nums) // 5
        nums = np.array(nums)
        nums = np.reshape(nums, (n_rows, 5))


        w_max = 0
        h_max = 0
        for row in range(n_rows):
            w, h = nums[row][3:]
            w_max = w if w > w_max else w_max
            h_max = h if h > h_max else h_max
        

        f.close()

    return w_max, h_max


def find_box_max_from_folder(folder):

    files = [os.path.join(folder, file) for file in os.listdir(folder) if 'txt' in file]
    w_max = 0
    h_max = 0
    for file in files:
        w, h = find_box_max_from_file(file)
        w_max = w if w > w_max else w_max
        h_max = h if h > h_max else h_max
    
    max = w_max if w_max > h_max else h_max
    print(f'Max bounding box is: {w_max, h_max}.\nSuggested shape for U-Net is: {max, max}')

    return max


def crop_obj(txt_folder, tiles_imgs_folder, save_imgs_folder, crop_shape = False):
    ''' Crops around detected object from yolo using a fixed dim. 
        NB output masks folders are gotten by replacing /images with /masks. '''

    # create output folders
    imgs_folder = tiles_imgs_folder
    folder = txt_folder
    if not os.path.isdir(save_imgs_folder):
        os.makedirs(save_imgs_folder)
    save_masks_folder = save_imgs_folder.replace('images', 'masks')
    if not os.path.isdir(save_masks_folder):
        os.makedirs(save_masks_folder)
    save_coords_folder = save_imgs_folder.replace('images', 'bb')
    if not os.path.isdir(save_coords_folder):
        os.makedirs(save_coords_folder)

    # get annotations
    if crop_shape is False: # i.e. equal to max glom found 
        max_box = find_box_max_from_folder(folder)
        print(f'Max box: {max_box}')
        shapes = [32/2048, 64/2048, 128/2048, 0.125, 0.25, 0.5, 1]
        closest = 100000
        for i, num in enumerate(shapes):
            closest = num if abs(max_box - num) < closest else closest
        crop_shape = closest
    else:
        crop_shape = crop_shape / 2048 # TODO: NB PERCHE' SIAMO IN % QUA, POI CAMBIA 2048
    print(f'Crop shape: {crop_shape * 2048}')
    files = [os.path.join(folder, file) for file in os.listdir(folder) if 'txt' in file]
    print(files)


    for i, file in enumerate(tqdm(files, desc = f'Cropping around {len(files)} gloms')):

        # read bounding box label
        with open(file, 'r') as f:
            text = f.read()
            f.close()

        # convert to np array
        text = text.replace('\n', ' ')
        text = text.split(sep = ' ')
        if text[-1] == ' ':
            text = text[:-1]
        if len(text) % 5 != 0: # remove '' at the end if not divisible by 0:
            del_idx = [i for i in range(len(text)) if text[i] == '']
            del_idx = del_idx[0]
            text.pop(del_idx)
        assert len(text) % 5 == 0, f'Text has object {len(text)}, not divisible by 5.'
        nums = [float(num) for num in text]
        n_rows = len(nums) // 5
        nums = np.array(nums)
        nums = np.reshape(nums, (n_rows, 5))

        # open corresponding image and mask tiles
        name = file.replace('/labels', '').replace('txt', 'png')
        img_p = os.path.join(imgs_folder, os.path.split(name)[1])
        mask_p = img_p.replace('images', 'masks')
        # print(f'Reading img: {img_p} and mask {mask_p}')
        mask = io.imread(mask_p)
        img = io.imread(img_p)
        w, h, _ = img.shape
        if i == 0:
            crop = int(w * crop_shape)
            # print(f'Cropping shape: {(crop, crop)}')


        name = name.replace('.', f'_glom-1.')
        for j, row in enumerate(range(n_rows)):
            xc, yc = nums[row][1:3]
            xc, yc = int(xc* (w-1)), int(yc* (h-1))

            # cropping coords (border edge problem)
            x_min = (xc - crop//2) if (xc - crop//2) >= 0 else 0
            if x_min != 0:
                x_max = (xc + crop//2) if (xc + crop//2) <= w else w
            else:
                x_max = crop
            if x_max == w:
                x_min = w - crop
            y_min = (yc - crop//2) if (yc - crop//2) >= 0 else 0
            if y_min != 0:
                y_max = (yc + crop//2) if (yc + crop//2) <= h else h
            else:
                y_max = crop
            if y_max == h:
                y_min = h - crop

            # print(f'Cropping coords: {x_min}:{x_max}, {y_min}:{y_max}')

            cropped_img = img[y_min:y_max, x_min:x_max, : ]
            cropped_mask = mask[y_min:y_max, x_min:x_max, : ]


            ## visual test
            # plt.figure()
            # plt.imshow(cropped_img)
            # plt.show()

            # save images and masks
            name = name.replace(f'_glom{j-1}', f'_glom{j}')
            img_fp = os.path.join(save_imgs_folder, os.path.split(name)[1])
            mask_fp = os.path.join(save_masks_folder, os.path.split(name)[1])
            coords_fp = os.path.join(save_coords_folder, os.path.split(name)[1].replace('png', 'txt'))
            io.imsave(fname = img_fp, arr = cropped_img, check_contrast=False)
            io.imsave(fname = mask_fp, arr = cropped_mask, check_contrast=False)
            save_txt = f'{str(y_min)},{str(y_max)},{str(x_min)},{str(x_max)}'
            with open(coords_fp, 'a') as f:
                f.write(save_txt)
                f.close()

    return


# if __name__ == '__main__':
#     from yolo_utils import get_last_detect
#     txt_folder = get_last_detect()
#     crop_obj(txt_folder= txt_folder, 
#             imgs_folder = '/Users/marco/hubmap/unet_data/images', 
#             save_imgs_folder = '/Users/marco/hubmap/unet_data/cropped/cropped/images',
#             crop_shape = False)