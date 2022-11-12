from array import ArrayType
from multiprocessing import current_process
from multiprocessing.dummy import Array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# Reminder : np.histogram returns an np.array

def grenander(segment: ArrayType, iteration: int, ascending: bool())-> ArrayType :
    """
    Computes the Grenander estimator, turning this portion of the histogram in a descending one

    Parameters:
    -----------
    `segment` : ArrayType 
        the segment that we use to compute its increasing Grenander estimator
    
    `iteration` : int
        the number of iterations that we desire for the algorithm.
    
    `ascending` : bool
        A boolean that indicates which estimator we are constructing.
        If False, it is the descending estimator (the one in the paper shown as an example)
        If True, it is the ascending estimator
        For the ascending estimator, we just have to apply the ascending algorithm to the opposite values.

    Returns:
    --------
    `histogram` : ArrayType
        Final histogram. The segmentation didn't change.

    """
    if(ascending):
        S = -segment
    else:
        S = segment

    for n in range(iteration):
        D = []
        pools = S.copy()
        i, L = 0, pools.size
        while i < L:
            j = i
            
            while(j < L-1 and pools[j] < pools[j+1]): #a possible increasing segment, not respecting the decreasing hypothesis
                j+=1
                
            if i==j : #we are in an increasing section or at the end, so the loop stopped at the first point
                D.append(pools[i]) #it remains unchanged
                i += 1 # we move away from that point
                
            else: #pool the violating sequence, the one that is decreasing
                pool = pools[i:j+1]
                D_r = np.sum(pool)/(j-i+1) #the constant value that this section is going to take
                for k in range(pool.size):
                    pool[k]=D_r
                D.extend(pool) #we add the modified segment to the output...
                i = j + 1 # we move away from that segment
        S = np.asarray(D)
    
    if(ascending):
        return -S
    return S

def test_stat(segment: ArrayType, estimator: ArrayType)-> bool :
    """
    Testing if the segment is close enough to its Grenander estimator

    Parameters:
    -----------
    `segment` : ArrayType 
        the segment that we use to compute its increasing Grenander estimator
    
    `estimator` : ArrayType
        the Grenander estimator (the law tested) of the segment 

    Returns:
    --------
    `test` : ArrayType
        result of the statistical test
        If 0, H0 is accepted : the increasing or decreasing hypothesis (does not matter here) is verified
        If 1, H0 is rejected : the hypothesis is rejected 
        (inconclusive...)
    """
    test = 0
    L = segment.size
    
    #exceptional case : if the segment's length is 2, 
    #H0 is accepted if and only if the estimator is the same as the segment
    if L<=2:
        return ~((segment==estimator).all())

    #construction of the statistic of the test
    T = 0
    for j in range(L):
        T += ((segment[j]-estimator[j])**2)/estimator[j]
    
    #test comparing T with the quantile of the chi-2 law
    alpha = 0.01 #standard level
    C = stats.chi2.ppf(1-alpha, L)
    
    if(T > C):
        test = 1
    return test

def is_unimodal(segment : ArrayType) -> bool :
    """
    Tests with a very simple condition if the segment is unimodal

    Parameters:
    -----------
    `segment` : array
        array of value whose unimodality must be determined


    Returns:
    --------
    `unimodality` : bool
        Returns True if the segment is unimodal.

    """
    max_index = 0
    max = segment[0]
    # find the maximum value
    for i in range(len(segment)-1) :
        if segment[i+1] > max :
            max_index = i+1
            max = segment[i+1]

    # check whether segment increases on [0, max_index]        
    increasing = True
    i = 0
    while(i < max_index) :
        # margin : variations of 1 don't impact monotony
        if(segment[i] <= segment[i+1] or np.abs(segment[i] - segment[i+1]) <=1) :
            i += 1
            pass
        else : 
            increasing = False
            break
    if(not increasing) :
            return False
    
    # check whether segment decreases on [max_index, len(segment)]
    i = max_index
    decreasing = True
    while( i < len(segment)-1) :
        # margin : variations of 1 don't impact monotony
        if(segment[i] >= segment[i+1] or np.abs(segment[i] - segment[i+1]) <=1) :
            i += 1
            pass
        else : 
            decreasing = False
            break
    if(not decreasing) :
        return False
    
    return True

def is_unimodal_new(segment : ArrayType) -> bool :
    """
    Tests with the statistical test if the segment has a mode.

    Parameters:
    -----------
    `segment` : array
        array of value whose unimodality must be determined


    Returns:
    --------
    `unimodality` : bool
        Returns True if the segment is unimodal.

    """
    L = segment.size
    #exceptions are dealt with in the other method, we will consider that L > 2 
    
    max_index = 0
    max = segment[0]
    # find the maximum value
    for i in range(L-1) :
        if segment[i+1] > max :
            max_index = i+1
            max = segment[i+1]
            
    #in the paper, they advise to go through the segment to find the mode
    #it's not always the maximum, the condition only states : "if c exists such that ..."
    #however, we can suppose that with ordinary histograms, a local maximum could be a mode
            
    S1 = segment[:max_index]
    S2 = segment[max_index:]
    grnd1_asc = grenander(S1,50,True)
    grnd1_dsc = grenander(S1,50,False)
    grnd2_asc = grenander(S2,50,True)
    grnd2_dsc = grenander(S2,50,False)
    if(test_stat(S1,grnd1_asc)==0 and test_stat(S2,grnd2_dsc)==0):
        print('max trouvé en : ',max_index)
        return True # segment[i] is considered a maximum, and the unimodal hypothesis is correct.
    if(test_stat(S1,grnd1_dsc)==0 and test_stat(S2,grnd2_asc)==0):
        print('min trouvé en :',max_index)
        return True # segment[i] is considered a minimum, and the unimodal hypothesis is correct.
    return False #if no correct max was found


def compute_finest_segmentation(histogram : ArrayType) -> ArrayType:
    """
    Computes the finest segmentation based on the local minima of the histogram.

    Parameters:
    -----------
    `histogram` : ArrayType
        array


    Returns:
    --------
    `segmentation` : ArrayType
        array of shape[nb_segments, 2] where the segments are each represented by their start and end index in histogram

    """

    segmentation = []
    segment_start_index = 0

    for i in range(1, len(histogram)-1) :
        previous, current, next = histogram[i-1], histogram[i], histogram[i+1] 
        # check if current is a local minimum
        if(previous >= current and current <= next) :
            segmentation.append(np.array([segment_start_index, i]))
            segment_start_index = i+1    

    segmentation.append(np.array([segment_start_index, len(histogram)-1])) #mustn't forget the final segment
    return segmentation


def merge_if_unimodal(histogram : ArrayType, segmentation : ArrayType, N : int) -> ArrayType:
    """
    Iterates through segmentation by considering n successive segments S1,..., Sn and checking if the union is unimodal. 
    If so, said segments are merged. If not, then S1 is added to the new segmentation and we consider S2,..., S(n+1).

    Parameters:
    -----------
    `histogram` : ArrayType 
        histogram
    `segmentation` : ArrayType
        array of the segments composing the histogram
    `N` : int
        integer in [2 , len(segmentation)] that determines the number of successive segments to consider for the merge

    Returns:
    --------
    `segmentation` : ArrayType
        Updated segmentation

    """
    # check lower bound
    if N < 2 : 
        print("N must be bigger than 2")
        return None

    start_index = 0
    end_index = 0
    iterator = 0 # corresponds to the number of initial segments that were added
    new_segmentation = []
    successive_failures = 0

    merge_possible = ( N <= len(segmentation)) # also makes sure N rsepects its upper bound

    while (merge_possible) :
        # consider N successive segments
        nb_possible_groups = len(segmentation) - N + 1


        while(iterator + N <= len(segmentation)) :
            start_index = segmentation[iterator][0]
            end_index = segmentation[iterator+N-1][1]
            merged_segment = np.array([start_index, end_index])

            if(len(histogram[start_index : end_index +1]) == 0) :
                print("debug : start_index and end_index computation should never permit to come here")
                successive_failures = nb_possible_groups
                break

            if(is_unimodal(histogram[start_index : end_index + 1])) :
            # segments can be merged
                iterator += N
                new_segmentation.append(merged_segment) # update the segmentation
                successive_failures = 0 
            else : 
                new_segmentation.append(segmentation[iterator])
                iterator += 1 
                successive_failures += 1

        # case when the number of segments left is < N : they must be all added one by one
        if(iterator != len(segmentation)) :
            for j in range(iterator, len(segmentation)) :
                new_segmentation.append(segmentation[j])

        segmentation = new_segmentation
        
        merge_possible = (successive_failures < nb_possible_groups) # false if all combinations of n successive segmentations can't be merged
        iterator = 0
        new_segmentation = []

    return np.asarray(segmentation)


def FTC(histogram : ArrayType) -> ArrayType :
    """
    Applies the FTC algorithm.

    Parameters:
    -----------
    `histogram` : ArrayType 
        histogram on which to apply the FTC algorithm

    Returns:
    --------
    `segmentation` : ArrayType
        Final segmentation respecting maximal unimodality criterion

    """

    segmentation = compute_finest_segmentation(histogram)

    N = 2
    N_in_range = (N <= len(segmentation))
    while(N_in_range) :
        print('the current segmentation is : ',np.asarray(segmentation))
        segmentation = merge_if_unimodal(histogram, segmentation, N)
        N += 1
        N_in_range = (N <= len(segmentation))
    return segmentation
    
#%% test

hist1 = np.array([0, 1, 2, 3, 4, 3, 2, 3, 0, 4, 5, 9, 4, 5, 4, 5, 3, 0, 7])
hist2 = np.array([0, 1, 2, 0, 3, 4, 5, 3, 6, 5, 6, 2, 5, 1, 3, 2, 0])
hist3 = np.array([40., 30., 25., 18., 22., 21., 14., 12., 9., 3., 6., 2., 1])
#hist4 is for testing that the test also works for the ascending hypothesis
hist4 = np.array([1., 2., 6., 3., 9., 12., 14., 21., 22., 18., 25., 30., 40]) 
hist5 = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 9., 6., 5., 4., 2., 1., 1., 2., 3., 4., 5., 6., 7., 8., 10., 9., 6., 5., 4., 2., 1.])

ftc1 = FTC(hist1)
print(ftc1)
ftc2 = FTC(hist2)
print(ftc2)
ftc5 = FTC(hist5)
print(ftc5)

grnd3 = grenander(hist3, 4, False)
print(grnd3)
grnd4 = grenander(hist4, 4, True)
print(grnd4)
test_3 = test_stat(hist3,grnd3)
print(test_3)
test_4 = test_stat(hist4,grnd4)
print(test_4)


#%%test with real images

from skimage import io as skio
import cv2

im = cv2.imread('ladybug.jpeg')
im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
imh = im[0]
N = imh.size
(histo,bins)=np.histogram(imh.reshape((-1,)),np.arange(0,256))
histo = histo
plt.plot(histo)
histo = histo.astype(float)
ftc_ladybug = FTC(histo)
print(ftc_ladybug)
for interval in ftc_ladybug:
    plt.axvline(x = interval[1], color = 'r')
    






