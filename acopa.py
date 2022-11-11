from ftc import *
import numpy as np
import cv2

# apply on hue histogram. 
#link each pixel from grey cylinder to interval

#%% Constants
Q = 360 # quantification factor


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

    HSI_image = RGB_image
    for y in range(len(RGB_image)):
        for x in range(len(RGB_image[0])): 
            pixel = RGB_image[y][x]
            R = pixel[0]
            G = pixel[1]
            B = pixel[2]
            I = (R + G + B) / 3
            S = np.sqrt((R - I)**2 + (G - I)**2 + (B - I)**2)
            if S > saturation_threshhold :
                H = np.arccos( ( (G - I) - (B - I)) / (S * np.sqrt(2)))
                HSI_image[y][x] = [H, S, I]
            else :
                HSI_image[y][x] = [-1, S, I]

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
    print("hues", hues)

    return FTC(np.histogram(hues, bins = [i/(2*np.pi) for i in range(Q+1)]))
            


def get_saturation_based_segmentation(hue_segmentation : ArrayType, HSI_image : ArrayType) ->ArrayType :
    """
    
    Parameters:
    ---------------

    Returns : 
    -------------
    Array where the nth element contains the saturation segmentation of the element in the nth segment of the hue segmentation
    """

    # group pixels based on the hue segmentation they belong to :
    # saturation_grouped_byehue[i] contains the pixels' saturation whose hues are in hue_segmentation[i]

    saturation_grouped_by_hue = [[] for i in len(hue_segmentation)] 

    for line_index in len(HSI_image) : 
        for row_index in len(HSI_image[0]) :
            pixel = HSI_image[line_index][row_index]
            hue_value = pixel[0]
            classified = False
            if hue_value == -1 : 
                saturation_grouped_by_hue[0].append(pixel[1])
                HSI_image[line_index][row_index][3] = 0 # this pixel is in the 0th segment of the hue segmentation
                classified = True
            i = 0
            while classified == False and i < len(hue_segmentation):
                segment = hue_segmentation[i]
                if hue_value >= segment[0] and hue_value <= segment[-1] : 
                    saturation_grouped_by_hue[i].append(pixel[1])
                    HSI_image[line_index][row_index][3] = i # this pixel is in the ith segment of the hue segmentation
                    classified = True
                i += 1
            if classified == False : 
                print("there was an issue while extracting the saturation of pixels grouped by hue.")
                exit

    # calculate the saturation segmentation 
    saturation_segmentation = []
    for S_i in saturation_grouped_by_hue : 
        saturation_segmentation.append([ FTC(np.histogram( S_i, bins = [i/Q for i in range(Q+1)] )) ]) #apply FTC on the pixels in S_i based on the saturation value

    return saturation_segmentation, HSI_image



def get_intensity_based_segmentation(saturation_segmentation : ArrayType, HSI_image ) -> ArrayType :
    """
    
    Parameters:
    ---------------

    Returns : 
    -------------
    The intensity_segmentation where intensity_segmentation[i][j] is the intensity segmentation for the pixels with
    their hue in hue_segmentation[i] and saturation_segmentation[i][j]. 
    """

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
                        if saturation_value >= subsegment[0] and saturation_value <= subsegment[-1] : 
                            intensity_grouped_by_saturation[i][j].append(pixel[1])
                            HSI_image[line_index][row_index][4] = j
                            classified = True
                    i += 1
                if classified == False : 
                    print("there was an issue while extracting the intensity of pixels grouped by saturation.")
                    exit

    # calculate the saturation segmentation 
    intensity_segmentation = []
    for i in range(len(intensity_grouped_by_saturation)) :
        S_i = intensity_grouped_by_saturation[i]
        for j in len(S_i) :
            S_ij = S_i[j]
            intensity_segmentation[i][j].append(FTC(np.histogram(S_ij, bins = [i/Q for i in range(Q+1)] )))

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
    return intensity_segmentation, HSI_image


#%% Mode computing functions
def compute_hue_modes(hue_segmentation : ArrayType) -> ArrayType : 
    hue_modes = []
    for S_i in hue_segmentation:
        hue_modes.append([np.max(S_i)])
    return hue_modes

def compute_saturation_modes(saturation_segmentation : ArrayType) -> ArrayType : 
    saturation_modes = []
    for S_i in saturation_segmentation : 
        sub_seg = []
        for S_ij in S_i :
            sub_seg.append([np.max(S_ij)])
        saturation_modes.append(sub_seg)
    return saturation_modes

def compute_intensity_modes(intensity_segmentation : ArrayType) -> ArrayType : 
    intensity_modes = []
    for S_i in intensity_segmentation : 
        sub_seg = []
        for S_ij in S_i :
            sub_sub_seg = []
            for S_ijk in S_ij : 
                sub_sub_seg.append([np.max(S_ijk)])
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
    hue_segmentation = get_hue_based_segmentation(HSI_image)
    saturation_segmentation = get_saturation_based_segmentation(hue_segmentation, HSI_image)
    intensity_segmentation = get_intensity_based_segmentation(saturation_segmentation, HSI_image)

    hue_modes = compute_hue_modes(hue_segmentation)
    saturation_modes = compute_saturation_modes(saturation_segmentation)
    intensity_modes = compute_intensity_modes(intensity_segmentation)

    new_image = np.zeros(image.shape)
    for row_index in range(len(image)):
        for column_index in range(len(image[0])):
            pixel = image[row_index][column_index]
            new_image[row_index][column_index][0] = hue_modes[pixel[3][0]]
            new_image[row_index][column_index][1] = saturation_modes[pixel[3]][pixel[4]][0]
            new_image[row_index][column_index][2] = intensity_modes[pixel[3]][pixel[4]][pixel[5]][0]
    return new_image

            


#%% Test

image =[[[200, 200, 200], [100, 150, 10], [30, 40, 180]], [[20, 200, 20], [100, 15, 10], [30, 140, 180]], [[200, 200, 200], [100, 150, 10], [30, 40, 180]]]

HSI_image = RGB_to_HSI_modified(image)
print(HSI_image)
hue_segmentation = get_hue_based_segmentation(HSI_image)
print(hue_segmentation)


img = cv2.imread()

