#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:26:47 2022

@author: ceciletillerot
"""

#%% SECTION 1 inclusion de packages externes 
import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
#%% SECTION 2 fonctions utiles pour le TP

def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP-2.10.app '
        endphrase=' ' 
    elif platform.system()=='Windows': 
        #ou windows ; probleme : il faut fermer gimp pour reprendre la main; 
        #si vous savez comment faire (commande start ?) je suis preneur 
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais pas comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'
    
    if normalise:
        m=im.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    
    nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)

def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI(defaut 0) et MAXI (defaut 255) seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP-2.10.app '
        endphrase= ' '
    elif platform.system()=='Windows': 
        #ou windows ; probleme : il faut fermer gimp pour reprendre la main; 
        #si vous savez comment faire (commande start ?) je suis preneur 
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase=' &'
    
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    
    nomfichier=tempfile.mktemp('TPIMA.pgm')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)

def quantize(im,n=2):
    """
    Renvoie une version quantifiee de l'image sur n (=2 par defaut) niveaux  
    """
    imt=np.float32(im.copy())
    if np.floor(n)!= n or n<2:
        raise Exception("La valeur de n n'est pas bonne dans quantize")
    else:
        m=imt.min()
        M=imt.max()
        imt=np.floor(n*((imt-m)/(M-m)))*(M-m)/n+m
        imt[imt==M]=M-(M-m)/n #cas des valeurs maximales
        return imt
    

def seuil(im,s):
    """ renvoie une image blanche(255) la ou im>=s et noire (0) ailleurs.
    """
    imt=np.float32(im.copy())
    mask=imt<s
    imt[mask]=0
    imt[~mask]=255
    return imt

#%% Simple median-cut method

k = 6 # le k dans le nombre 2^k des couleurs sur la palette
im=skio.imread('ladybug.jpeg') #une image au choix

def final_image(im, im_simple):
    r_average = np.mean(im_simple[:,0])
    g_average = np.mean(im_simple[:,1])
    b_average = np.mean(im_simple[:,2])
    
    for pixel in im_simple:
        im[pixel[3],pixel[4]] = [r_average, g_average, b_average]
     
def split_image(im, im_temp, k):
    if len(im_temp) == 0: # on peut arriver à une liste vide
        return 
        
    if k == 0: # on arrive à la fin : il faut associer à l'image les bonnes couleurs
        final_image(im, im_temp)
        return
    
    #choix du channel
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
        
    im_temp = im_temp[im_temp[:,channel].argsort()] # sort par rapport au channel intéressant : on ne touche pas aux indices des pixels !!!
    median = int((len(im_temp)+1)/2) # séparer chaque liste en 2
    split_image(im, im_temp[0:median], k-1)
    split_image(im, im_temp[median:], k-1)

im2 = []
for i in range(len(im)):
    for j in range(len(im[0])):
        pixel = im[i,j]
        im2.append([pixel[0], pixel[1], pixel[2], i, j])
im2 = np.array(im2) 

# on travaille avec une image qui est d'un autre format : chaque pixel est une liste : couleurs + indexs 
# on a plus le format contraignant de l'image, avec les pixels au bon endroit, mais une image avec des éléments qu'on peut bouger

split_image(im, im2, k)
viewimage(im)

#%% Conversion images
import cv2

image = cv2.imread('girl.jpeg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
labImage = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)
cv2.imshow('Lab image', labImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Algorithmes histogrammes


def finest_seg(histo):
    S = [[histo[0]]]
    L = 0
    N = len(histo)
    k = 1 #le début de notre prochain segment
    for i in range(1, N-1):
        if (histo[i-1]>histo[i]) and (histo[i+1]>histo[i]): #minima local trouvé
            S.append([histo[k:i]])
            k = i+1
            L+=1
    if k != N:
        S.append([histo[k:N-1]])
    S.append([histo[N]])
    return L, np.array(S)

L, S = finest_seg(h)
while
def()



        












