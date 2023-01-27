import os 
folder = '/Users/marco/Downloads/muw_slides/images'
folder_2 = '/Users/marco/Downloads/try_train'
tiles = [file for file in os.listdir(folder) if 'png' in file and 'DS' not in file]
slides = [file.split('_')[0] for file in tiles]
unique_slides = list(set(slides))
slides_n = len(unique_slides)
print(unique_slides)
print(slides_n)

unique_fnames = list(set([file.split('.')[0] for file in os.listdir(folder_2)]))
# del_files = [os.path.join(folder_2, file) for file in os.listdir(folder_2) for slide in unique_slides if slide not in file]
print(unique_fnames)
raise Exception
for del_file in del_files:
    try:
        os.remove(del_file)
    except: 
        print(f"❌ {del_file}")