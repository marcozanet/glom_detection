import numpy as np 
from sklearn.cluster import KMeans
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import slic, mark_boundaries

def kmeans_seg_image(img: np.array, k: int  = 2, location = False ):
    """ Segments the image by using k-means clustering. 
        If location is True, it also concatenates X and Y coords as additional channels (tot. = 5 ch)"""



    print(img.shape)
    io.imshow(img)

    W, H, C = img.shape 
    if location is True:
        hey = np.zeros(shape = (W, H, 2))
        for i in range(W):
            for j in range(H):
                hey[i, j, :] = [i, j]
        img = np.concatenate((img, hey), axis = 2)
        W, H, C = np.shape(img) # i.e. now C = 5


    # if location is True:
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             for k in range(img.shape[2]):
    #     img = np.expand_dims(img, 3)
    #     print(img.shape)
    #     print(img)

    # raise NotImplementedError()
    img = np.reshape(img, (-1, C))
    n_triplets = img.shape[0]
    img = np.float32(img)
    print(img.shape)
    model = KMeans(n_clusters= k)
    idxs = model.fit_predict(img)
    print(idxs.shape)
    img = np.reshape(idxs, (W, H, 1))
    io.imshow(img)



    return


def slic_seg_image(img: np.array, mask: np.array):
    """ Segments the image by using SLIC clustering. """

    print(img.shape)
    io.imshow(img)
    # io.imshow(mask)
    mask = mask > 0

    print(np.unique(mask))
    print(mask.shape)
    segments_slic = slic(img, n_segments=40, compactness=20, sigma=1, mask = mask)
    io.imshow(mark_boundaries(img, segments_slic))

    


    return

def test_slic_seg_image():
    fp = '/Users/marco/hubmap/training/train/model_train/unet/images/8242609fa_6_4_glom4.png'
    img = io.imread(fp)
    mask = io.imread(fp.replace('images', 'masks'))
    slic_seg_image(img, mask)

    return

def test_kmeans_seg_image():
    fp = '/Users/marco/hubmap/training/train/model_train/unet/images/8242609fa_6_4_glom4.png'
    img = io.imread(fp)
    kmeans_seg_image(img, k = 3,  location = False)

    return


if __name__ == "__main__":
    test_slic_seg_image()
    # test_kmeans_seg_image()