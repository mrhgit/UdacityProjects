#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:59:15 2018

@author: mach1uvnc
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, glob

print ("Imported Custom Lane Lines Presentation Code\n")

def multi_graph(images,titles=None,big=False):
    if big:
        height = 6
    else:
        height = 3
    fig = plt.figure(figsize=(15,height),dpi=80)
    n = len(images)
    
    for idx,img in enumerate(images):
        fig.add_subplot(1,n,idx+1)
        if len(img.shape)==2:
            plt.imshow(img,cmap='gray')
        else:
            plt.imshow(img)
        if titles and titles[idx]:
            plt.title(titles[idx])
    return

def convert_flat_to_rgb(img):
    if len(img.shape) > 2:
        return img
    if np.max(img) < 1.1:
        out_img = np.asarray(np.dstack((img,img,img))*255,dtype='uint8')
    else:
        out_img = np.asarray(np.dstack((img,img,img)),dtype='uint8')
    return out_img

def quadrants(img1,img2,img3,img4):
    
    img1_rgb = convert_flat_to_rgb(img1)
    img2_rgb = convert_flat_to_rgb(img2)
    img3_rgb = convert_flat_to_rgb(img3)
    img4_rgb = convert_flat_to_rgb(img4)

    img = np.zeros_like(img1_rgb)
    
    img1_resized = cv2.resize(img1_rgb, (0,0), fx=0.5, fy=0.5)
    img2_resized = cv2.resize(img2_rgb, (0,0), fx=0.5, fy=0.5)
    img3_resized = cv2.resize(img3_rgb, (0,0), fx=0.5, fy=0.5)
    img4_resized = cv2.resize(img4_rgb, (0,0), fx=0.5, fy=0.5)
    
    img[:img.shape[0]//2,:img.shape[1]//2,:] += img1_resized
    img[img.shape[0]//2:,:img.shape[1]//2,:] += img2_resized
    img[:img.shape[0]//2,img.shape[1]//2:,:] += img3_resized
    img[img.shape[0]//2:,img.shape[1]//2:,:] += img4_resized
    
    return img
