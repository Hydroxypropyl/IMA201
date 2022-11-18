#%% imports 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#%% Simple median-cut method

k = 4 # the parameter of the median-cut method : there are 2^k colors on the palette
K = 2**k # the number of colors in the palette, ie the parameter of the K-mean method
im=skio.imread('dalhia.jpg') #une image au choix

def final_image(im, im_simple):
    r_average = np.mean(im_simple[:,0])
    g_average = np.mean(im_simple[:,1])
    b_average = np.mean(im_simple[:,2])
    
    for pixel in im_simple:
        im[pixel[3],pixel[4]] = [r_average, g_average, b_average]
     
def split_image(im, im_temp, k):
    
    if len(im_temp) == 0:
        return 
        
    if k == 0: # at the end, we fill the image with the right colors
        final_image(im, im_temp)
        return
    
    #choice of the channel
    red = im_temp[:,0]
    green = im_temp[:,1]
    blue = im_temp[:,2]
    
    range_red = red.max() - red.min()
    range_green = green.max() - green.min()
    range_blue = blue.max() - blue.min()
    channel = 0
    if range_green > range_blue and range_green > range_red:
        channel = 1
    if range_blue > range_green and range_blue > range_red:
        channel = 2
        
    im_temp = im_temp[im_temp[:,channel].argsort()] # sort according to the appropraite channel
    median = int((len(im_temp)+1)/2) # divide the list into two parts
    split_image(im, im_temp[0:median], k-1)
    split_image(im, im_temp[median:], k-1)


def median_cut(im, k):
    # create a picture where pixel also contain the information of their index
    im2 = []
    for i in range(len(im)):
        for j in range(len(im[0])):
            pixel = im[i,j]
            im2.append([pixel[0], pixel[1], pixel[2], i, j])
    im2 = np.array(im2) 
    split_image(im,im2,k)


#%% K-means method

def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

def K_means(im, k): # k is nb_colors
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    ima = np.array(im, dtype=np.float64) / 255
    
    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(ima.shape)
    assert d == 3
    image_array = np.reshape(ima, (w * h, d))
    
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array_sample)
    
    # Get labels for all points
    labels = kmeans.predict(image_array)
    
    return recreate_image(kmeans.cluster_centers_, labels, w, h)

#%% Tests
    
median_cut(im,k)
plt.imshow(im)
plt.title(label = "k = " + str(k) +", nb colors = " + str(2**k))
plt.imshow(K_means(im,K))
plt.title(label = "nb colors = " + str(2**k))
