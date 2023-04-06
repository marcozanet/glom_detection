from normalizer import Normalizer



def test_Normalizer(): 

    print("####### TEST normalizer: ⏳ ##########")
    import sys 
    system = 'mac' if sys.platform == 'darwin' else 'windows'

    target_path = '/Users/marco/helical_tests/test_stainnormalizer/target_pas.png'
    to_transform_path = '/Users/marco/helical_tests/test_normalizer_new'
    verbose = True 
    show = True
    save_folder = None
    replace_images = True
    normalizer = Normalizer(target_path=target_path, to_transform_path=to_transform_path, 
                            verbose = verbose, show = show, save_folder=save_folder, replace_images=replace_images)
    normalizer()
    print("####### TEST normalizer: ✅ ##########")

    return


if __name__ == '__main__': 
    test_Normalizer()