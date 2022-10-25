from array import ArrayType
from multiprocessing import current_process
from multiprocessing.dummy import Array
import numpy as np

# Reminder : np.histogram returns an np.array


def is_unimodal(segment : ArrayType) -> bool :
    """
    Computes the finest segmentation based on the local minima of the histogram.

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

    return segmentation


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
        segmentation = merge_if_unimodal(histogram, segmentation, N)
        N += 1
        N_in_range = (N <= len(segmentation))
    return segmentation

### test
hist1 = np.array([0, 1, 2, 3, 4, 3, 2, 3, 0, 4, 5, 9, 4, 5, 4, 5, 3, 0, 7])
hist2 = np.array([0, 1, 2, 0, 3, 4, 5, 3, 6, 5, 6, 2, 5, 1, 3, 2, 0])

ftc1 = FTC(hist1)
ftc2 = FTC(hist2)
print(hist1)
print(ftc1)
print(hist2)
print(ftc2)


