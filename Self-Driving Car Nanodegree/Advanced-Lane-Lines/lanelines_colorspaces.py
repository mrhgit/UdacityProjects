#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:55:32 2018

@author: mach1uvnc
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, glob

print ("Imported Custom Lane Lines Colorspaces Code\n")

def split_img(img):
    return [np.squeeze(x) for x in np.split(img,img.shape[2],axis=2)]

def get_cmyk(img):
    one_layer = np.zeros_like(img[:,:,0])
    cmyk = np.asarray(np.dstack((one_layer,one_layer,one_layer,one_layer)),dtype='float64')
    img1 = img/255.0
    r = img1[:,:,0]
    g = img1[:,:,1]
    b = img1[:,:,2]
    k = np.min(1.0-img1,axis=2)
    cmyk[:,:,0] = (1 - r - k) / (1 - k + 1e-5)
    cmyk[:,:,1] = (1 - g - k) / (1 - k + 1e-5)
    cmyk[:,:,2] = (1 - b - k) / (1 - k + 1e-5)
    cmyk[:,:,3] = k
    cmyk *= 255.0
    return np.asarray(cmyk,dtype='uint8')

 # Colorspace Conversion functions
def get_hsv(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

def get_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def get_gray(img, chan=-1):
    if len(img.shape) > 2:
        if chan==-1:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            return np.copy(img[:,:,chan])
    else:
        return np.copy(img)

def norm_to_255(img,row_height=-1):
    img_out = np.copy(img)
    if len(img.shape)==2:
        img_out = np.expand_dims(img_out,axis=2)
        
    for ch in range(img_out.shape[2]):
        if row_height==-1:
            ch_min = np.min(img_out[:,:,ch])*1.
            ch_max = np.max(img_out[:,:,ch])*1.
            img_out[:,:,ch] = (img_out[:,:,ch] - ch_min) / (ch_max-ch_min) * 255.
        else:
            for r in range(0,img_out.shape[0],row_height):
                r_start = r
                r_end = r + row_height
                ch_min = np.min(img_out[r_start:r_end,:,ch])*1.
                ch_max = np.max(img_out[r_start:r_end,:,ch])*1.
                img_out[r_start:r_end,:,ch] = (img_out[r_start:r_end,:,ch] - ch_min) / (ch_max-ch_min) * 255.

        
    img_out = np.squeeze(img_out)
    return np.asarray(img_out,dtype='uint8')
    
# Gradient functions
def simple_sobel(img, orient='x', chan=-1, ksize=3):
    gray = get_gray(img,chan)
    return cv2.Sobel(gray, cv2.CV_64F, int(orient=='x'), int(orient=='y'), ksize=ksize)

def get_thresh(img, thresh_min=0, thresh_max=255, chan=-1):
    gray = get_gray(img,chan)
    binary_output = np.zeros_like(gray)
    binary_output[(gray >= thresh_min) & (gray <= thresh_max)] = 1
    return np.asarray(binary_output,dtype='uint8') # type conversion to work with binary logic

def get_sobel(img, func='abs', chan=-1, thresh_min=0, thresh_max=255, ksize=3):
    sobelx = simple_sobel(img, 'x', chan, ksize=ksize)
    sobely = simple_sobel(img, 'y', chan, ksize=ksize)
    if func == 'abs':
        sobel = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
    elif func == 'absx':
        sobel = np.absolute(sobelx)
    elif func == 'absy':
        sobel = np.absolute(sobely)
    elif func == 'atan':
        sobel = np.absolute(np.arctan2(np.absolute(sobely),np.absolute(sobelx)))
    elif func == 'absxydelta':
        sobel = np.absolute(sobelx - sobely)
        
    if func != 'atan':
        scaled_sobel = np.uint8(255.*sobel / np.max(sobel))
        return get_thresh(scaled_sobel,thresh_min,thresh_max)
    else:
        scaled_sobel = sobel
        return get_thresh(scaled_sobel,thresh_min,thresh_max)
        
