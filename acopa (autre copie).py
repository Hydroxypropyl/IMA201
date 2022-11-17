#%% imports
from ftc_with_statistical_test import *
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans

# apply on hue histogram. 
#link each pixel from grey cylinder to interval

#%% Constants
Q = 9 # quantification factor
Q_hue = 9
#%% convert image to a modified HSI

def HSI_to_RGB_colors(HSI_colors) :
    eps = 0.001
    RGB_colors = []
    for color in HSI_colors:
        H = color[0]
        S = color[1]
        I = color[2]

        if np.abs(H) <= eps :
            R = I + 2*I*S
            G = I - I*S
            B = I - I*S
        elif np.abs(H-2*np.pi/3) < eps:
            R = I - I*S
            G = I + 2*I*S
            B = I - I*S
        elif np.abs(H + 2*np.pi/3) < eps : 
            R = I - I*S
            G = I - I*S
            B = I + 2*I*S
        elif eps < H and H < 2*np.pi/3  :
            R = I + I*S*np.cos(H)/np.cos(np.pi/3 - H)
            G = I + I*S*(1 - np.cos(H)/np.cos(np.pi/3 - H))
            B = I - I*S
        elif np.abs(H) > 2*np.pi/3 :
            R = I - I*S
            G = I + I*S*np.cos(H-2*np.pi/3)/np.cos(np.pi-H)
            B = I + I*S*(1 - np.cos(H-2*np.pi/3)/np.cos(np.pi-H)) 
        elif -2*np.pi/3 < H and H < eps : 
            R = I + I*S*(1 - np.cos(H-4*np.pi/3)/np.cos(5*np.pi/3-H))
            G = I - I*S
            B = I + I*S*np.cos(H-4*np.pi/3)/np.cos(5*np.pi/3-H)
        RGB_colors.append([math.floor(R),math.floor(G),math.floor(B)])
    return RGB_colors

def HSI_to_RGB(HSI_image : ArrayType) -> ArrayType :
    print("entering HSI_to_RGB...")
    eps = 0.001
    RGB_image = []
    for row_index in range(len(HSI_image)):
        row = []
        for column_index in range(len(HSI_image[row_index])):
            B = 0
            G = 0
            R = 0
            pixel = HSI_image[row_index][column_index]
            H = pixel[0]
            S = pixel[1]
            I = pixel[2]

            if np.abs(H) <= eps :
                R = I + 2*I*S
                G = I - I*S
                B = I - I*S
            elif np.abs(H-2*np.pi/3) < eps:
                R = I - I*S
                G = I + 2*I*S
                B = I - I*S
            elif np.abs(H + 2*np.pi/3) < eps : 
                R = I - I*S
                G = I - I*S
                B = I + 2*I*S
            elif eps < H and H < 2*np.pi/3  :
                R = I + I*S*np.cos(H)/np.cos(np.pi/3 - H)
                G = I + I*S*(1 - np.cos(H)/np.cos(np.pi/3 - H))
                B = I - I*S
            elif np.abs(H) > 2*np.pi/3 :
                R = I - I*S
                G = I + I*S*np.cos(H-2*np.pi/3)/np.cos(np.pi-H)
                B = I + I*S*(1 - np.cos(H-2*np.pi/3)/np.cos(np.pi-H)) 
            elif -2*np.pi/3 < H and H < eps : 
                R = I + I*S*(1 - np.cos(H-4*np.pi/3)/np.cos(5*np.pi/3-H))
                G = I - I*S
                B = I + I*S*np.cos(H-4*np.pi/3)/np.cos(5*np.pi/3-H)
            row.append([math.floor(R),math.floor(G),math.floor(B)])
        RGB_image.append(row)
    return RGB_image



def BGR_to_modified_HSI(BGR_image : ArrayType, saturation_threshhold : int = 0.08) -> ArrayType:
    """
    Takes the image in BGR and returns the image where each pixel is coded by [H, S, I, 0, 0, 0]]

    Parameters:
    ---------------
    `image` :
    Initial image in BGR

    `saturation_threshhold` :
    Saturation threshhold under which a pixel is considered achromatic, so part of the grey cylinder

    Returns : 
    -------------
    `HSI_image`:
    The image converted to HSI with three additional 0 coordinates for each pixel
    """
    print("entering BGR_to_HSI...")
    HSI_image = np.zeros((BGR_image.shape[0], BGR_image.shape[1], 6))
    for y in range(len(BGR_image)):
        for x in range(len(BGR_image[y])): 
            pixel = BGR_image[y][x]
            #récupérer les coordonnées RGB : opencv donne du BGR attention
            B = int(pixel[0])
            G = int(pixel[1])
            R = int(pixel[2])
            I = (R + G + B) / 3 # in [0, 255]
            if I == 0 :
                S = 0
            else : 
                S = 1 - np.min([B, G, R])/I #in [0, 1]
            if True :
                alpha = (2*R-G-B)/(2*255)
                beta = np.sqrt(3)*(G-B)/(2*255)
                H = np.arctan2(beta, alpha) # H is in [-pi, pi]
                HSI_image[y][x] = [H, S, I, 0, 0, 0]
            else :
                HSI_image[y][x] = [-1, S, I, 0, 0, 0]

    return HSI_image
    

#%% Different segmentation steps

def get_hue_based_segmentation(HSI_image : ArrayType)  -> ArrayType :
    """
    Returns the segmentation and the histogram of the hue component of  `HSI_image`

    Parameters:
    ---------------
    `image` : the image outputed by BGR_to_modified_HSI

    Returns : 
    -------------
    `FTC` :
    The segmentation of the hue histogram where the achromatic pixels haven't been taken into account

    `histo` :
    The hue histogram
    """
    print("entering get_hue_based_segmentation...")
    hues = []
    for line in HSI_image:
        for pixel in line : 
            if True : #condition of chromaticity
                hues.append(pixel[0])
    histo, bins = np.histogram(hues, bins = np.arange(-np.pi, np.pi, step=2*np.pi/(Q_hue+1)) ) #H is better quantified, and is in [0, pi]
    return FTC(histo), histo
            


def get_saturation_based_segmentation(hue_segmentation : ArrayType, hue_histogram : ArrayType, HSI_image : ArrayType) -> ArrayType :
    """
    Calculates the saturation based segmentation of the different groups of pixesl defined by the hue segmentation

    Parameters:
    ---------------
    `hue_segmentation`:
    The hue segmentation of `HSI_image`

    `HSI_image`:
    An image in modified HSI just as BGR_to_modified_HSI_outputs (each pixel is [H, S, I, i, j, k])

    Returns : 
    -------------
    `saturation segmentation`:
    segmentations of the saturation histograms based on the groups determined by the hue_segmentation

    `HSI_image` :
    updated image with the i component of pixels updated

    `saturation_histograms` :
    [i] contains the  saturation histogram for pixels in S_i in hue_histogram 
   
    """
    print("entering get_saturation_based_segmentation...")
    # group pixels based on the hue segmentation they belong to :
    # saturation_grouped_by_hue[i] contains the pixels' saturation whose hues are in hue_segmentation[i]

    saturation_grouped_by_hue = [[] for i in range(len(hue_segmentation))] 

    for line_index in range(len(HSI_image)) : 
        for row_index in range(len(HSI_image[line_index])) :
            pixel = HSI_image[line_index][row_index]
            hue_value = pixel[0]
            classified = False
            #go through all the segments to find the one pixel belongs to, based on hue    
            i = 0
            while classified == False and i < len(hue_segmentation):
                segment = hue_segmentation[i] #indices of start and end in hue_histogram
                #print("intervals,", -np.pi+ segment[0]*2*np.pi/Q, hue_value, -np.pi +(segment[1]+1)*2*np.pi/Q)
                if hue_value >=(-np.pi+ segment[0]*2*np.pi/Q_hue) and hue_value <= (-np.pi +(segment[1]+1)*2*np.pi/Q_hue) + 0.00001: 
                    saturation_grouped_by_hue[i].append(pixel[1])
                    HSI_image[line_index][row_index][3] = i # this pixel is in the ith segment of the hue segmentation
                    classified = True
                i += 1
            if classified == False : 
                print("intervals,", -np.pi+ segment[0]*2*np.pi/Q_hue, hue_value, -np.pi +(segment[1]+1)*2*np.pi/Q_hue)
                print("there was an issue while extracting the saturation of pixels grouped by hue.")
                exit(1)

    # calculate the saturation segmentation 
    saturation_segmentation = []
    saturation_histograms = []
    for S_i in saturation_grouped_by_hue : 
        histo, bins = np.histogram(S_i, bins = [i/Q_sat for i in range(Q_sat+1)]) #quantify S when calculating histogram
        saturation_histograms.append(histo)
        saturation_segmentation.append(FTC(histo)) #apply FTC on the pixels in S_i based on the saturation value

    return saturation_segmentation, HSI_image, saturation_histograms



def get_intensity_based_segmentation(saturation_segmentation : ArrayType, saturation_histograms : ArrayType, HSI_image ) -> ArrayType :
    """
    
    Parameters:
    ---------------
    `saturation_segmentation`:

    `saturation_histograms` :


    Returns : 
    -------------
    The intensity_segmentation where intensity_segmentation[i][j] is the intensity segmentation for the pixels with
    their hue in hue_segmentation[i] and saturation_segmentation[i][j]. 
    """
    print("entering get_intensity_based_segmentation...")
    #create empty groups with same shape as saturation_segmentation
    intensity_grouped_by_saturation = [[] for i in range(len(saturation_segmentation))]
    intensity_segmentation = [[] for i in range(len(saturation_segmentation))]
    intensity_histograms = [[] for i in range(len(saturation_segmentation))]
    for i in range(len(saturation_segmentation)):
        nb_sub_segments = len(saturation_segmentation[i])
        intensity_grouped_by_saturation[i] =[[] for j in range(nb_sub_segments)]
        intensity_segmentation[i] =[[] for j in range(nb_sub_segments)]
        intensity_histograms[i] =[[] for j in range(nb_sub_segments)]

    # classify based on saturation value
    for line_index in range(len(HSI_image)) : 
        for row_index in range(len(HSI_image[0])) :
            pixel = HSI_image[line_index][row_index]
            saturation_value = pixel[1]
            classified = False
            i = 0
            while classified == False and i < len(saturation_segmentation):
                if math.floor(pixel[3]) == i : #the pixel must be in the ith segment of the hue segmentation
                    segment = saturation_segmentation[i]
                    for j in range(len(segment)) :
                        if j== 0:
                            min = segment[j][0]/Q_sat
                        else : 
                            min = segment[j-1][-1]/Q_sat
                        max = segment[j][-1]/Q_sat
                        if j == len(segment)-1 :
                            max = (segment[j][-1] + 1) /Q_sat
                        #print("interval :", min, saturation_value, max)
                        if saturation_value >= min and saturation_value <= max : 
                            intensity_grouped_by_saturation[i][j].append(pixel[2])
                            HSI_image[line_index][row_index][4] = j
                            classified = True
                            break
                i += 1
            if classified == False : 
                print("there was an issue while extracting the intensity of pixels grouped by saturation :", line_index, row_index)
                exit(1)

    # calculate the intensity segmentation 
    for i in range(len(intensity_grouped_by_saturation)) :
        S_i = intensity_grouped_by_saturation[i]
        for j in range(len(S_i)) :
            S_ij = S_i[j]
            #print("S_" +str(i) +"," +str(j) +" : ", S_ij)
            histo, bins = np.histogram(S_ij, bins = np.arange(0, 255, 255/(Q_int+1)))
            intensity_segmentation[i][j] = FTC(histo)
            intensity_histograms[i][j] = histo

    #update the intensity segment index in HSI_image 
    for line_index in range(len(HSI_image)):
        for row_index in range(len(HSI_image[0])):
            pixel = HSI_image[line_index][row_index]
            S_ij = intensity_segmentation[math.floor(pixel[3])][math.floor(pixel[4])]
            for k in range(len(S_ij)) :
                S_ijk = S_ij[k]
                intensity = pixel[2]
                #print("intervals :", S_ijk[0]*255/Q, intensity, S_ijk[-1]*255/Q)
                if intensity >= S_ijk[0]*255/Q_int and intensity <=S_ijk[-1]*255/Q_int :
                    HSI_image[line_index][row_index][5] = k
                    break
    return intensity_segmentation, HSI_image, intensity_histograms


#%% Mode computing functions
def compute_hue_modes(hue_segmentation : ArrayType, hue_histogram : ArrayType) -> ArrayType : 
    print("entering computing_hue_modes...")
    hue_modes = []
    for S_i in hue_segmentation:
        segment = hue_histogram[S_i[0] : S_i[1] + 1]
        max_index = np.where( segment == np.amax(segment))[0] + S_i[0]
        hue_modes.append(-np.pi + max_index*2*np.pi/Q_hue)
    return hue_modes

def compute_saturation_modes(saturation_segmentation : ArrayType, saturation_histograms : ArrayType) -> ArrayType : 
    print("entering computing_saturation_modes...")
    saturation_modes = []
    for i in range(len(saturation_segmentation)) : 
        sub_seg = []
        for j in range(len(saturation_segmentation[i])) :
            index_min = saturation_segmentation[i][j][0]
            index_max = saturation_segmentation[i][j][1]
            segment = saturation_histograms[i][index_min : index_max+1]
            max_index = np.where( segment == np.amax(segment))[0] + index_min
            sub_seg.append(max_index/Q_sat)
        saturation_modes.append(sub_seg)
    return saturation_modes

def compute_intensity_modes(intensity_segmentation : ArrayType, intensity_histograms : ArrayType) -> ArrayType : 
    """
    Parameters : 
    ---------------
    `intensity_segmentation` : array 
    The intensity segmentation of the image
    
    `intensity_histograms` : array
    The intensity histograms of all the segments
    
    Returns : 
    -------------
    `intensity_modes`: array
    Same shape as intensity_segmentation, containing in each segment a unique value corresponding to the mode.

    `nb_modes` : int 
    The total number of modes of the intensity component 
    """
    print("entering computing_intensity_modes...")
    intensity_modes = []
    nb_modes = 0
    for i in range(len(intensity_segmentation)) : 
        sub_seg = []
        for j in range(len(intensity_segmentation[i])) :
            sub_sub_seg = []
            for k in range(len(intensity_segmentation[i][j])) : 
                S_ijk = intensity_segmentation[i][j][k]
                segment = intensity_histograms[i][j][S_ijk[0]: S_ijk[1]+1]
                max_index = np.where( segment == np.amax(segment))[0] + S_ijk[0]           
                sub_sub_seg.append(max_index*255/Q_int)
                nb_modes +=1
            sub_seg.append(sub_sub_seg)
        intensity_modes.append(sub_seg)
    return intensity_modes, nb_modes




#%% Putting everything together

def acopa(image : ArrayType, HSI : bool = False) -> ArrayType :
    """
    Applies the acopa algorithm on `image`. 
    By default, `image` must be in BGR and is converted in the function.
    If `HSI` = True then the `image` is considered already in HSI and acopa is applied without any conversion.
    Parameters:
    --------------
    `image` : array
    The image to apply the acopa algorithm

    `HSI` : bool
    Whether the image is coded in HSI or not. 
    Returns : 
    ---------
    """
    print("entering acopa...")
    if not(HSI) :
        HSI_image = BGR_to_modified_HSI(image)
    else : 
        HSI_image = image
    hue_segmentation, hue_histogram = get_hue_based_segmentation(HSI_image)
    saturation_segmentation, HSI_image, saturation_histograms = get_saturation_based_segmentation(hue_segmentation, hue_histogram, HSI_image)
    intensity_segmentation, HSI_image, intensity_histograms = get_intensity_based_segmentation(saturation_segmentation, saturation_histograms, HSI_image)
    hue_modes = compute_hue_modes(hue_segmentation, hue_histogram)
    saturation_modes = compute_saturation_modes(saturation_segmentation, saturation_histograms)
    intensity_modes, nb_colors = compute_intensity_modes(intensity_segmentation, intensity_histograms)
    

    colors = []
    for i in range(len(hue_modes)) :
        H = hue_modes[i][0]
        for j in range(len(saturation_modes[i])) :
            S = saturation_modes[i][j][0]
            for k in range(len(intensity_modes[i][j])):
                I = intensity_modes[i][j][k][0]
                colors.append([H, S, I])

    return colors, nb_colors

            
def final_palette(img, seeds): 
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))
    KMeans=kmeans(n_clusters = len(seeds), init = seeds).fit(img_array)
    rgb_cols = KMeans.cluster_centers_.round(0).astype(int)
    labels = KMeans.fit_predict(img_array)
    img_quant = np.reshape(rgb_cols[labels],(w,h,d))
    return img_quant

#%% Test
for name in ["glady.jpg"] :
    image = cv2.imread(name)
    HSI_image = BGR_to_modified_HSI(image)
    for i in [50] :
        for j in [50] :
            Q_sat = i
            Q_int = 50
            Q_hue = j
            colors, nb_colors = acopa(HSI_image, True)
            colors = HSI_to_RGB_colors(colors)
            img1 = final_palette(image, colors)
            plt.imshow(img1)
            plt.title(label = "Q_int = " +str(Q_int) +" Q_sat = " + str(Q_sat) + " Q_hue = "+ str(Q_hue) + ", nb colors = " +str(nb_colors))
            plt.show()


