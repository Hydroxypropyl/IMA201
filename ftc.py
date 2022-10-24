from multiprocessing import current_process
import numpy as np

# Reminder : np.histogram returns an np.array

hist = np.array([0, 1, 2, 3, 4, 3, 2, 3, 0, 4, 5, 9, 4, 5, 4, 5, 3, 0, 7])

def is_unimodal(segment) :
    max_index = 0
    max = segment[0]
    for i in range(len(segment)-1) :
        if segment[i+1] > max :
            max_index = i+1
            max = segment[i+1]
    increasing = True
    print("max index :", max_index)
    i = 0
    while(i < max_index) :
        if(segment[i] <= segment[i+1] or np.abs(segment[i] - segment[i+1]) <=1) :
            i += 1
            pass
        else : 
            increasing = False
            print("increasing", increasing, "i", segment[i], "i+1", segment[i+1])
            break
    if(not increasing) :
            return False
    
    i = max_index
    decreasing = True
    while( i < len(segment)-1) :
        if(segment[i] >= segment[i+1] or np.abs(segment[i] - segment[i+1]) <=1) :
            i += 1
            pass
        else : 
            decreasing = False
            print("decreasing", decreasing)
            break
    if(not decreasing) :
        return False
    
    print("increasing", increasing, "decreasing", decreasing)
    return True

def compute_finest_segmentation(histogram) :
    """
    Computes the finest segmentation based on the local minima of the histogram.

    Parameters:
    -----------
    `histogram` : np.array


    Returns:
    --------
    `segmentation` : np.array of shape[n, 2] where the n segments are each represented by their start and end index in histogram

    """

    segmentation = []
    segment_start_index = 0
    for i in range(1, len(histogram)-1) :
        previous, current, next = histogram[i-1], histogram[i], histogram[i+1] 
        #check if current is a local minimum
        if(previous >= current and current <= next) :
            segmentation.append(np.array([segment_start_index, i]))
            segment_start_index = i+1        
    segmentation.append(np.array([segment_start_index, len(histogram)-1])) #mustn't forget the final segment
    return segmentation


def merge_if_unimodal(histogram, segmentation, N) :
    """
    Iterates through segmentation by considering n successive segments S1,..., Sn and checking if the union is unimodal. 
    If so, said segments are merged. If not, then S1 is added to the new segmentation and we consider S2,..., S(n+1).

    Parameters:
    -----------
    `histogram` : histogram
    `segmentation` : array of the segments composing the histogram
    `N` : integer in [2 , len(segmentation)] that determines the number of successive segments to consider for the merge

    Returns:
    --------
    `segmentation` : Updated segmentation

    """

    iterator = 0 #corresponds to the number of initial segments that were added
    new_segmentation = []
    segmentation_length = len(segmentation)
    merge_possible = True
    successive_failures = 0 

    if N < 2 : 
        print("N must be bigger than 2")
        return None

    start_index = 0
    end_index = 0
    while (merge_possible) :
        new_segmentation = []
        # consider N successive segments
        nb_possible_groups = len(segmentation) - N + 1
        #print("N = ", N, "len seg :", len(segmentation), "iterator :", iterator)

        while(iterator + N <= len(segmentation)) :
            print("len seg :", len(segmentation))
            print("iterator =", iterator, "N =", N)
            start_index = segmentation[iterator][0]
            end_index = segmentation[iterator+N-1][1]
            #print("start",start_index)
            #print("end", end_index)
            merged_segment = np.array([start_index, end_index])
            print(histogram[start_index : end_index + 1])
            if(len(histogram[start_index : end_index +1]) == 0) :
                print("test")
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

        #case when the number of segments left is < N : they must be all added one by one
        if(iterator != len(segmentation)) :
            for j in range(iterator, len(segmentation)) :
                new_segmentation.append(segmentation[j])
        segmentation = new_segmentation
        # returns false if all combinations of n successive segmentations can't be merged
        merge_possible = (successive_failures < nb_possible_groups)
        iterator = 0
    return segmentation


def FTC(histogram) :
    segmentation = compute_finest_segmentation(histogram = histogram)
    for N in range(2, len(segmentation)) :
        segmentation = merge_if_unimodal(histogram, segmentation, N)
        print("N = ", N, "segmentation : ", segmentation)
    return segmentation

ftc = FTC(hist)
print(ftc)


