#%% imports
from ftc_with_statistical_test import *
import numpy as np
import cv2

# apply on hue histogram. 
#link each pixel from grey cylinder to interval

#%% Constants
Q = 50 # quantification factor


#%% convert image to a modified HSI

def RGB_to_HSI_modified(RGB_image : ArrayType, saturation_threshhold : int = Q/(2*np.pi)) -> ArrayType and ArrayType:
    """
    
    Parameters:
    ---------------
    `image` :
    `saturation_threshhold` : int
    Saturation threshhold under which a pixel is considered achromatic, so part of the grey cylinder
    Returns : 
    -------------
    Image where the pixels are [H, S, I, 0, 0, 0]
    """
    HSI_image = np.zeros((RGB_image.shape[0], RGB_image.shape[1], 6))
    for y in range(len(RGB_image)):
        for x in range(len(RGB_image[0])): 
            pixel = RGB_image[y][x]
            B = pixel[0]
            G = pixel[1]
            R = pixel[2]
            I = (R + G + B) / 3
            S = np.sqrt((R - I)**2 + (G - I)**2 + (B - I)**2)
            if S > saturation_threshhold :
                H = np.arccos( ( (G - I) - (B - I) ) / (S * np.sqrt(2))) #T H est dans [0, pi]
                HSI_image[y][x] = [H, S, I, 0, 0, 0]
            else :
                HSI_image[y][x] = [-1, S, I, 0, 0, 0]

    return HSI_image
    

#%% Different segmentation steps

def get_hue_based_segmentation(HSI_image : ArrayType)  -> ArrayType :
    """
    
    Parameters:
    ---------------
    `image` : 

    Returns : 
    -------------
    The segmentation of the hue histogram where the achromatic pixels haven't been taken into account
    """
    hues = []
    for line in HSI_image:
        for pixel in line : 
            if pixel[0] != -1 : #condition of chromaticity
                hues.append(pixel[0])
    #print("hues", hues)
    histo, bins = np.histogram(hues, bins = [(i*2*np.pi/Q) for i in range(Q+1)])
    return FTC(histo), histo
            


def get_saturation_based_segmentation(hue_segmentation : ArrayType, hue_histogram : ArrayType, HSI_image : ArrayType) -> ArrayType :
    """
     Array where the nth element contains the saturation segmentation of the element in the nth segment of the hue segmentation
    Parameters:
    ---------------

    Returns : 
    -------------
    `saturation segmentation`:
    segmentations of the saturation histograms based on the groups determined by the hue_segmentation
    `HSI_image` :
    updated image
    `saturation_histograms` :
    [i] contains the  saturation histogram for pixels in S_i in hue_histogram 
   
    """

    # group pixels based on the hue segmentation they belong to :
    # saturation_grouped_byehue[i] contains the pixels' saturation whose hues are in hue_segmentation[i]

    saturation_grouped_by_hue = [[] for i in range(len(hue_segmentation))] 

    for line_index in range(len(HSI_image)) : 
        for row_index in range(len(HSI_image[line_index])) :
            pixel = HSI_image[line_index][row_index]
            hue_value = pixel[0]
            classified = False
            if hue_value == -1 : #pixel is achromatic
                saturation_grouped_by_hue[0].append(pixel[1])
                HSI_image[line_index][row_index][3] = 0 # this pixel is in the 0th segment of the hue segmentation
                classified = True
            i = 0
            while classified == False and i < len(hue_segmentation):
                segment = hue_segmentation[i] #indices of start and end in hue_histogram
                #print("intervals,", hue_histogram[segment[0]], hue_value, hue_histogram[segment[1]])
                if hue_value >= segment[0]*2*np.pi/Q and hue_value <= segment[1]*2*np.pi/Q: 
                    saturation_grouped_by_hue[i].append(pixel[1])
                    HSI_image[line_index][row_index][3] = i # this pixel is in the ith segment of the hue segmentation
                    classified = True
                i += 1
            if classified == False : 
                print("there was an issue while extracting the saturation of pixels grouped by hue.")
                exit(1)

    # calculate the saturation segmentation 
    saturation_segmentation = []
    saturation_histograms = []
    for S_i in saturation_grouped_by_hue : 
        histo = np.histogram( S_i)
        saturation_histograms.append(histo)
        saturation_segmentation.append([ FTC(histo) ]) #apply FTC on the pixels in S_i based on the saturation value

    return saturation_segmentation, HSI_image, saturation_histograms



def get_intensity_based_segmentation(saturation_segmentation : ArrayType, saturation_histograms : ArrayType, HSI_image ) -> ArrayType :
    """
    
    Parameters:
    ---------------

    Returns : 
    -------------
    The intensity_segmentation where intensity_segmentation[i][j] is the intensity segmentation for the pixels with
    their hue in hue_segmentation[i] and saturation_segmentation[i][j]. 
    """
    print(saturation_segmentation)
    #create empty groups with same shape as saturation_segmentation
    intensity_grouped_by_saturation = [[] for i in range(len(saturation_segmentation))]
    for i in range(len(saturation_segmentation)):
        nb_sub_segments = len(saturation_segmentation[i])
        intensity_grouped_by_saturation[i].append([] for j in range(nb_sub_segments))
    
    # classify based on saturation value
        for line_index in range(len(HSI_image)) : 
            for row_index in range(len(HSI_image[0])) :
                pixel = HSI_image[line_index][row_index]
                saturation_value = pixel[1]
                classified = False
                i = 0
                while classified == False and i < len(saturation_segmentation):
                    segment = saturation_segmentation[i]
                    for j in range(len(segment)) :
                        subsegment = segment[j] 
                        print(len(subsegment[0]))
                        if saturation_value >= subsegment[0]/Q and saturation_value <= subsegment[-1]/Q : 
                            intensity_grouped_by_saturation[i][j].append(pixel[1])
                            HSI_image[line_index][row_index][4] = j
                            classified = True
                    i += 1
                if classified == False : 
                    print("faulty pixel :", line_index, row_index)
                    print("there was an issue while extracting the intensity of pixels grouped by saturation.")
                    exit

    # calculate the intensity segmentation 

    intensity_segmentation = [[[]for j in range(len(saturation_segmentation[i]))] for i in range(len(saturation_segmentation))]
    intensity_histograms = intensity_segmentation
    for i in range(len(intensity_grouped_by_saturation)) :
        S_i = intensity_grouped_by_saturation[i]
        for j in len(S_i) :
            S_ij = S_i[j]
            histo = np.histogram(S_ij, bins = [i/Q for i in range(Q+1)] )
            intensity_segmentation[i][j].append(FTC(histo))
            intensity_histograms[i][j] = histo

    #update the intensity segment index in HSI_image 
    for line_index in range(len(HSI_image)):
        for row_index in range(len(HSI_image[0])):
            pixel = HSI_image[line_index][row_index]
            S_ij = intensity_segmentation[pixel[3]][pixel[4]]
            for k in range(len(S_ij)) :
                S_ijk = S_ij[k]
                intensity = pixel[2]
                if intensity >= S_ijk[0] and intensity <=S_ijk[-1] :
                    HSI_image[line_index][row_index][5] = k
    return intensity_segmentation, HSI_image, intensity_histograms


#%% Mode computing functions
def compute_hue_modes(hue_segmentation : ArrayType, hue_histogram : ArrayType) -> ArrayType : 
    hue_modes = []
    for S_i in hue_segmentation:
        hue_modes.append([np.max(hue_segmentation[S_i[0], S_i[1]])])
    return hue_modes

def compute_saturation_modes(saturation_segmentation : ArrayType, saturation_histograms : ArrayType) -> ArrayType : 
    saturation_modes = []
    for i in range(len(saturation_histograms)) : 
        sub_seg = []
        for j in range(len(saturation_histograms[i])) :
            index_min = saturation_segmentation[i][j][0]
            index_max = saturation_segmentation[i][j][1]
            sub_seg.append([np.max(saturation_histograms[i][index_min : index_max+1])])
        saturation_modes.append(sub_seg)
    return saturation_modes

def compute_intensity_modes(intensity_segmentation : ArrayType, intensity_histograms : ArrayType) -> ArrayType : 
    intensity_modes = []
    for i in range(len(intensity_segmentation)) : 
        sub_seg = []
        for j in range(len(intensity_segmentation[i])) :
            sub_sub_seg = []
            for k in range(len(intensity_segmentation[i][j])) : 
                S_ijk = intensity_segmentation[i][j][k]
                sub_sub_seg.append([np.max(intensity_histograms[i][j][S_ijk[0]: S_ijk[1]+1])])
            sub_seg.append(sub_sub_seg)
        intensity_modes.append(sub_seg)
    return intensity_modes




#%% Putting everything together

def acopa(image : ArrayType) -> ArrayType :
    """
    Parameters:
    --------------
    
    Returns : 
    ---------
    """
    HSI_image = RGB_to_HSI_modified(image)
    hue_segmentation, hue_histogram = get_hue_based_segmentation(HSI_image)
    saturation_segmentation, HSI_image, saturation_histograms = get_saturation_based_segmentation(hue_segmentation, hue_histogram, HSI_image)
    intensity_segmentation, HSI_image, intensity_histograms = get_intensity_based_segmentation(saturation_segmentation, saturation_histograms, HSI_image)

    hue_modes = compute_hue_modes(hue_segmentation, hue_histogram)
    saturation_modes = compute_saturation_modes(saturation_segmentation, saturation_histograms)
    intensity_modes = compute_intensity_modes(intensity_segmentation, intensity_histograms)

    new_image = np.zeros(image.shape)
    for row_index in range(len(image)):
        for column_index in range(len(image[0])):
            pixel = image[row_index][column_index]
            new_image[row_index][column_index][0] = hue_modes[pixel[3][0]]
            new_image[row_index][column_index][1] = saturation_modes[pixel[3]][pixel[4]][0]
            new_image[row_index][column_index][2] = intensity_modes[pixel[3]][pixel[4]][pixel[5]][0]
    return new_image

            


#%% Test
image = cv2.imread("ladybug.jpeg")
HSI_image = RGB_to_HSI_modified(image)
hue_segmentation, hue_histogram = get_hue_based_segmentation(HSI_image)
print(hue_histogram)
saturation_segmentation, HSI_image, saturation_histograms = get_saturation_based_segmentation(hue_segmentation, hue_histogram, HSI_image)







# %%
