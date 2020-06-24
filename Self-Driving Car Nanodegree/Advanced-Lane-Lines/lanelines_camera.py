#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 18:30:05 2018

@author: mach1uvnc
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os, glob

print ("Imported Custom Lane Lines Camera Code\n")
class PERSPECTIVE:
    def __init__(self,old_pts,new_pts):
        self.M = cv2.getPerspectiveTransform(old_pts,new_pts)
        self.Minv = cv2.getPerspectiveTransform(new_pts,old_pts)
        
    def warp(self,img):
        img_size = (img.shape[1],img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
    
    def unwarp(self,img):
        img_size = (img.shape[1],img.shape[0])
        return cv2.warpPerspective(img, self.Minv, img_size)
 

class CAMERA_CALIBRATION:
    def __init__(self,n_corners):
        # Fancy way to make object points (0,0,0),(1,0,0),(2,0,0)...(0,1,0),(1,1,0),(2,1,0)...
        self.objp = np.zeros((np.prod(n_corners),3),np.float32)
        self.objp[:,:2] = np.mgrid[0:n_corners[0],0:n_corners[1]].T.reshape(-1,2)
        
        self.objpoints = [] # 3-D, real-world object points
        self.imgpoints = [] # 2-D, image plane points
        self.imgsize = None
        self.n_corners = n_corners
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.successful_cal = False

    def calibrate_image(self,img,to_gray=cv2.COLOR_RGB2GRAY):
        gray = cv2.cvtColor(img,to_gray)
        ret, corners = cv2.findChessboardCorners(gray, self.n_corners,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FILTER_QUADS+cv2.CALIB_CB_FAST_CHECK)
        if not ret:
            # Couldn't find it - return a red image
            new_img = np.copy(img)
            new_img[:,:,1:3] = 0
            new_img[:,:,0] = gray
        else:
            new_img = cv2.drawChessboardCorners(img, self.n_corners, corners, ret)
        return new_img, ret, corners

    def feed_image(self,img):
        self.imgsize = (img.shape[1],img.shape[0])
        
        cal_image, successful_cal, corners = self.calibrate_image(img)
            
        if successful_cal:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            self.successful_cal = True
            self.recalibrate_camera()
        return cal_image, successful_cal, corners
    
    def recalibrate_camera(self):
        if self.successful_cal:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints,self.imgpoints,self.imgsize, None, None)
            
    def undistort(self,img):
        if self.successful_cal:
            undist = cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
            return undist
        else:
            return None

    def get_objpoints(self):
        return self.objpoints
    
    def get_imgpoints(self):
        return self.imgpoints
    
    def get_imgsize(self):
        return self.imgsize