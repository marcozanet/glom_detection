import os 
from glob import glob
from tqdm import tqdm

def normalize_labels(root:str, tile_size:int):

    folds = ['train', 'val', 'test']
    for fold in folds:
        # print(os.path.join(root, fold, 'labels', '*.txt'))
        labels = glob(os.path.join(root, fold, 'labels', '*.txt'))
        print(f"Normalizing {fold}")
        # print(len(labels))
        for i,label in enumerate(tqdm(labels)):
            normalize_label(label_fp=label, tile_size=tile_size)
            # if i == 10:
            #     raise NotImplementedError()


    return

def normalize_label(label_fp:str, tile_size:int):

    assert os.path.isfile(label_fp)
    with open(label_fp, 'r') as f:
        rows = f.readlines()
    
    # print(rows)

    new_rows = []
    for row in rows:
        # print('rpova')
        # print(row)
        # print(row[-2:])
        row = row.replace('\n', '')
        items = row.split(' ')
        class_n = items[0]
        pos = items[1:]
        # print(items)
        # print(pos)
        # print('fsjdf')
        # print('sdkjfb')

        pos = [f"{str(int(el)/tile_size) } "for el in pos]
        # print(pos)

        new_pos = ''.join(str(x) for x in pos)
        new_row = f"{class_n} " + new_pos # secondo me non ci va + '\n'
        # print(new_row)
        # print('fdlnfdsl')
        new_rows.append(new_row)
    # print(new_rows)

    # overwrite
    with open(label_fp, 'w') as f:
        f.writelines(new_rows)


    # raise NotImplementedError()




    return


if __name__ == '__main__':
    root = r'D:\marco\hubmap_slides\detection\tiles'
    tile_size = 2048
    normalize_labels(root=root, tile_size=tile_size)