#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 03:06:23 2018

@author: mach1uvnc
"""
import numpy as np
import lesson_functions as lf
from scipy.ndimage.measurements import label


class VEHICLE_TRACKER:
    def __init__(self,minheat,search_params,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial_size, hist_bins,
                            color_space,hog_channel,spatial_feat, hist_feat, hog_feat):
        self.minheat = minheat
        self.search_params = search_params
        self.svc = svc
        self.X_scaler = X_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.color_space = color_space
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.reset()
    
    def reset(self):
        self.heatmap = None
        self.imgsize = None
    
    def feed_image(self,img,isVideo=True):
        draw_image1 = np.copy(img)
        draw_image2 = np.copy(img)
        
        if self.imgsize==None or not isVideo:
            self.imgsize = img.shape[0:2]
            self.heatmap = np.zeros(self.imgsize,dtype='float64')

        # Detect Cars
        hot_windows = []
        for search_param in self.search_params:
            xstart = search_param["xstart"]
            xstop = search_param["xstop"]
            ystart = search_param["ystart"]
            ystop = search_param["ystop"]
            scale = search_param["scale"]
            hot_windows += lf.find_cars(img, xstart,xstop,ystart, ystop, scale, self.svc, self.X_scaler, self.orient,
                                self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins,
                                color_space=self.color_space,hog_channel=self.hog_channel,spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            
        window_img = lf.draw_boxes(draw_image1, hot_windows, color=(0, 0, 255), thick=6)  

        # Add to heatmap
        if not isVideo:
            self.heatmap = lf.add_heat(self.heatmap,hot_windows)
            self.heatmap_thresh = lf.apply_threshold(self.heatmap, self.minheat)
        else:
            self.heatmap = np.clip(self.heatmap-1,0,self.minheat)
            self.heatmap = lf.add_heat(self.heatmap,hot_windows)
            self.heatmap_thresh = lf.apply_threshold(self.heatmap, self.minheat)
            self.heatmap += self.heatmap_thresh

        labels = label(self.heatmap_thresh)
        label_boxes = lf.get_labeled_bboxes(labels)

        heatmap_img = np.asarray(np.dstack([self.heatmap,self.heatmap,self.heatmap])/np.max(self.heatmap)*255,dtype='uint8')

        out_img = lf.draw_boxes(draw_image2, label_boxes, color=(0, 255, 0), thick=6)  
        results_dict = {"window_img":window_img,"out_img":out_img,"heat_map":heatmap_img,"heat_map_thresh":self.heatmap_thresh}
        
        return results_dict
