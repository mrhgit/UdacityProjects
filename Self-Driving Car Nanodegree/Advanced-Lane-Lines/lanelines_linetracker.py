#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 23:57:17 2018

@author: mach1uvnc
"""
import numpy as np
import cv2

class LINE():
    # LINE Object Detects and Keeps Track of Lines
    # Performs cold and warm searches
    def __init__(self,which_line="left"):
        self.reset()
        self.search_start = 380 # where to look for line in cold-start mode - left line
        self.search_end = 580 # where to look for line in cold-start mode - left line
        self.which_line = which_line.lower()
        if which_line != "left":
            self.search_start = 620 # where to look for line in cold-start mode - right line
            self.search_end = 940 # where to look for line in cold-start mode - right line
        self.undetected_max = 20
        self.pairline = None
        
    def set_pairline(self,pairline):
        # We can use information from a pair line to help us determine if our curve is good
        # or to help us perform a warm search
        self.pairline = pairline

    def reset(self):
        self.detected = False # valid line detected previously?
        self.fit = None # quadratic polyfit result
        self.radius_of_curvature = None # radius of curvature
        self.offset = None # offset from road center
        self.fitx = None # all points used in polyfit (x coordinates)
        self.fity = None # all points used in polyfit (y coordinates)
        self.window_pts = None
        self.undetected_count = 0
            
    def find_hist_peak(self,img):
        # Find a peak in the lane line image histogram
        histogram = 1.0*np.sum(img[img.shape[0]//2:,:], axis=0) # look in lower half of image
        return np.argmax(histogram[self.search_start:self.search_end])+self.search_start

    def calc_curvature(self,y_eval=719, mx=3.7/320,my=3.0/150):
        # Calculate the radius of curvature of a quadratic polynomial
        y_eval *= my
        
        self.fit_scaled = np.copy(self.fit)
        self.fit_scaled[0] *= mx/(my**2)
        self.fit_scaled[1] *= mx/my
        
        A = self.fit_scaled[0]
        B = self.fit_scaled[1]
        
        df1 = 2*A*y_eval + B
        df2 = 2*A
        
        R = (1 + df1**2)**(3./2) / abs(df2)
        self.radius_of_curvature = R
    
        return self.radius_of_curvature
    
    def fit_y(self,y_eval,fit=None):
        # Evaluate the fitted polynomial at a specific input value
        if fit==None:
            fit = self.fit
        yvec = [y_eval**2, y_eval, 1]
        return np.dot(fit,yvec)
    
    def calc_offset(self,y_eval=719,img_width=1280,my=3.0/150, mx=3.7/320):
        # Calculate the offset in meters
        bottom_intercept = self.fit_y(y_eval)
        ctr = img_width/2. - 0.5
        self.offset = (bottom_intercept - ctr) * mx
        return self.offset

    def cold_lane_search(self,imgs, nwindows=20, margin=15, minpix=30, x_current=None):
        badfit = True
        for img in imgs:
            # Attempts to return a set of points representing the lane line
            lane_inds = []
    
            # Given the number of windows, calculate their height
            window_height = np.int(img.shape[0]/nwindows)
    
            # Find all points that passed the gradient thresholding in the image
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
    
            # Initialize points using a histogram technique
            if x_current==None:
                x_current = self.find_hist_peak(img)
    
            extra_margin = 0
            self.window_pts = []
            valid_windows = []
            for window in range(nwindows):
                if window==0:
                    extra_margin = 2*margin
                # Calculate the search window boundaries based on the current window number
                win_y_low = img.shape[0] - (window+1)*window_height
                win_y_high = win_y_low + window_height
                win_x_low = x_current - (margin+extra_margin)
                win_x_high = x_current + (margin+extra_margin) #* 2
                self.window_pts += [((win_x_low,win_y_low),(win_x_high,win_y_high))]
    
                # Grab the points
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                    (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
    
                # Append these indices to the lists
                lane_inds.append(good_inds)
    
                # If you found > minpix pixels, recenter next window on their mean position
                #print ("window %d rt %d lft %d" % (window,len(good_left_inds),len(good_right_inds)))
                if len(good_inds) > minpix:
                    x_current = np.int(np.mean(nonzerox[good_inds]))
                    valid_windows += [window]
    
                if window==0:
                    extra_margin = 0
                extra_margin += 1 + window % 2
    
                #if window==5: break
    
            # Concatenate the arrays of indices
            lane_inds = np.concatenate(lane_inds)
    
            if len(valid_windows) >= 2:
                if max(valid_windows)-min(valid_windows)>=2:
                    # Extract pixel positions
                    fitx = nonzerox[lane_inds]
                    fity = nonzeroy[lane_inds]
                    
                    # Fit a second order polynomial to each
                    fit = np.polyfit(fity, fitx, 2)
                    
                    if self.curve_ok(fit,x_current):
                        self.fitx = fitx
                        self.fity = fity
                        self.fit = fit
                        self.calc_curvature()
                        self.calc_offset()
                        self.detected = True
                        self.undetected_count = 0
                        badfit = False
                        break
            
        if badfit:
            self.detected = False
            self.undetected_count += 1
            self.fitx = None
            self.fity = None
            
        return self.fit
    
    def warm_lane_search(self, imgs, fit=None, nwindows=20, margin=20, minpix=30, usePair=False):  
        if usePair:
            if self.pairline.which_line=="left":
                x_current = self.pairline.fit_y(719) + 320
            else:
                x_current = self.pairline.fit_y(719) - 320
            x_current = int(x_current)
                
            return self.cold_lane_search(imgs,nwindows,margin,minpix,x_current)

        if fit is None:
            fit = self.fit

        badfit = True
        for img in imgs:
            
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + 
                fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + 
                fit[1]*nonzeroy + fit[2] + margin))) 
        
            # Extract pixel positions
            fitx = nonzerox[lane_inds]
            fity = nonzeroy[lane_inds] 
    
            if len(fity) > 3:
                #print(len(nonzeroy),len(lane_inds),len(fity),len(fitx))
                if max(fity)-min(fity)>=50:
                    # Fit a second order polynomial to each
                    fit = np.polyfit(fity, fitx, 2)
                    
                    if self.curve_ok(fit):
                        self.fitx = fitx
                        self.fity = fity
                        self.fit = fit
                        self.calc_curvature()
                        self.calc_offset()
                        self.detected = True
                        self.undetected_count = 0
                        badfit = False
                        break
            
        if badfit:
            self.detected = False
            self.undetected_count += 1
            self.fitx = None
            self.fity = None
            
        return self.fit

    def curve_ok(self,fit=None,start_x=None):
        if fit==None:
            # If we're testing our current fit
            fit = self.fit
            bottom_intercept = self.fit_y(719,fit)
            return (bottom_intercept > self.search_start) and (bottom_intercept < self.search_end)
        else:
            if start_x is not None: # are we comparing to a histogram start?
                bottom_intercept = self.fit_y(719,fit)
                diff = abs(bottom_intercept - start_x)
                return (diff < 100)
            else: # we can make sure the bottom intercept is within a reasonable range
                if self.fit==None:
                    bottom_intercept = self.fit_y(719,fit)
                    return (bottom_intercept > self.search_start) and (bottom_intercept < self.search_end)
                else:
                    # we're testing a new fit compared to our current one
                    bottom_intercept_new = self.fit_y(719,fit)
                    bottom_intercept_old = self.fit_y(719,self.fit)
                    diff = abs(bottom_intercept_new - bottom_intercept_old)
                    return (diff < 100)

    def still_warm(self):
        have_a_fit = self.fit != None
        have_recent_detection = self.undetected_count < self.undetected_max
        
        return have_a_fit and have_recent_detection

    def feed_lc_images(self,imgs,warm=False):
        if not warm:
            self.reset()
        
        if self.still_warm():
            self.warm_lane_search(imgs)
        else:
            if self.pairline is not None and self.pairline.still_warm():
                self.warm_lane_search(imgs,usePair=True)
            #print(self.which_line,"COLD",self.undetected_count)
            self.cold_lane_search(imgs)

class SMART_FIT:
    # SMART_FIT keeps tracks of two lines and probably needs to be replaced with the
    #  pairline functionality of the regular LINE class...
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.left_fit = None
        self.right_fit = None
        self.reset_after_undetected = 20
        
        
    def calc_curvature(self,fit,y_eval=719, mx=3.7/320,my=3.0/150):
        y_eval *= my
        
        fit_scaled = np.copy(fit)
        fit_scaled[0] *= mx/(my**2)
        fit_scaled[1] *= mx/my
        
        A = fit_scaled[0]
        B = fit_scaled[1]
        
        df1 = 2*A*y_eval + B
        df2 = 2*A
        
        R = (1 + df1**2)**(3./2) / abs(df2)
    
        return R

    def get_curvature(self):
        #return (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2
        if self.left_fit==None or self.right_fit==None:
            return 0
        return (self.calc_curvature(self.left_fit) + self.calc_curvature(self.right_fit))/2
        
    def calc_offset(self,fit,y_eval=719,img_width=1280,my=3.0/150, mx=3.7/320):
        if self.left_fit==None or self.right_fit==None:
            return 0
        yvec = [y_eval**2, y_eval, 1]
        bottom_intercept = np.dot(fit,yvec)
        ctr = img_width/2. - 0.5
        offset = (bottom_intercept - ctr) * mx
        return offset
    
    def get_offset(self):
        return self.calc_offset(self.right_fit) + self.calc_offset(self.left_fit)
    
    def check_closeness(self,fit1,fit2):
        if fit1==None or fit2==None:
            return False
        ratio_diff = abs(fit2 / fit1 - 1.0)
        #print("ratio diff",ratio_diff,fit1,fit2)
        return np.all(ratio_diff <= 0.33)

    def merge_fits(self,fit1,fit2,alpha=0.75,merge=False):
        if fit2==None:
            return fit1
        if fit1==None or merge==False:
            return fit2
        
        if self.check_closeness(fit1,fit2):
            merged = fit2
        else:
            merged = fit1 * alpha + fit2 * (1 - alpha)
        return merged

    def feed_left_fit(self,fit,merge=False):
        self.left_fit = self.merge_fits(self.left_fit,fit,merge=merge)
 
    def feed_right_fit(self,fit,merge=False):
        self.right_fit = self.merge_fits(self.right_fit,fit,merge=merge)
        
    def get_fits(self):
        return self.left_fit,self.right_fit

class LANELINES:
    # Takes in the camera calibration, perspective transform and functions to find lane colors in images
    #  Is fed frames of video or single images
    #  returns a dictionary containing debug info and images and the image overlaid with lane and info
    def __init__(self,camera_cal,persp_txfr,lanecolor_detect_ptrs):
        self.left_line = LINE("left")
        self.right_line = LINE("right")
        self.left_line.set_pairline(self.right_line)
        self.right_line.set_pairline(self.left_line)
        self.smart_fit = SMART_FIT()
        
        self.camera_cal = camera_cal
        self.persp_txfr = persp_txfr
        self.lanecolor_detect_ptrs = lanecolor_detect_ptrs
        
    def reset(self):
        self.left_line.reset()
        self.right_line.reset()
        self.smart_fit.reset()
        
    def combine_images(self,img1,img2,alpha1=1,alpha2=0.3):
        return cv2.addWeighted(img1, alpha1, img2, alpha2, 0)
    
    def if_addTo(self,img,addTo=False):
        if addTo:
            return np.copy(img)
        else:
            return np.zeros_like(img).astype(np.uint8)
        
    def if_unwarp(self,img,unwarp=False):
        if unwarp:
            return self.persp_txfr.unwarp(img)
        else:
            return img
        
    def draw_detected_lines(self,img,showBoxes=False,addTo=False,unwarp=False):
        color_warp = self.if_addTo(img,addTo)

        if showBoxes:
            if self.left_line.window_pts != None:
                for w in self.left_line.window_pts:
                    cv2.rectangle(color_warp,w[0],w[1],(0,255,0), 2)
            if self.right_line.window_pts != None:
                for w in self.right_line.window_pts:
                    cv2.rectangle(color_warp,w[0],w[1],(0,255,255), 2)

        if self.left_line.detected:
            color_warp[self.left_line.fity,self.left_line.fitx] = [255,0,0]
        if self.right_line.detected:
            color_warp[self.right_line.fity,self.right_line.fitx] = [0,0,255]

        return self.if_unwarp(color_warp,unwarp)
            
    def draw_lane(self,img,showPolys=False,addTo=False,unwarp=False):
        color_warp = self.if_addTo(img,addTo)

        left_fit,right_fit = self.smart_fit.get_fits()
        if left_fit==None or right_fit==None:
            return color_warp
        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
        pts = np.hstack((pts_left,pts_right))
              
        cv2.fillPoly(color_warp, np.int_([pts]),(0,32,0))
        if showPolys:
            cv2.polylines(color_warp,np.int_([pts_left]),False,(255,255,0),4)
            cv2.polylines(color_warp,np.int_([pts_right]),False,(255,255,0),4)

        return self.if_unwarp(color_warp,unwarp)

    def shadow_text(self,img,text,location,color=(255,255,255),scale=1.5,width=4,font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.putText(img,text, (location[0]-2,location[1]+2),font,scale,(0,0,0),width+2*2)
        cv2.putText(img,text,location,font,scale,color,width)
        return
    
    def get_curvature(self):
        return self.smart_fit.get_curvature()
    
    def get_offset(self):
        return self.smart_fit.get_offset()

    def draw_info(self,img,addTo=True):
        unwarped = self.if_addTo(img,addTo)
        
        # Add radius of curvature
        curvature = np.min([self.get_curvature()/1000.,9.99])
        self.shadow_text(unwarped,"Radius of Curvature: %2.2f km" % curvature,(10,60))
        
        # Add offset from center
        self.shadow_text(unwarped,"Offset from Center: %0.2f m" % self.get_offset(),(10,120))
        
        return unwarped
   
    def generate_debug_img(self,img):
        out_img = np.asarray(np.dstack((img,img,img)),dtype='uint8')
        lines_img = self.draw_detected_lines(out_img,showBoxes=True,addTo=True,unwarp=False)
        lane_img = self.draw_lane(lines_img,showPolys=False,addTo=False,unwarp=False)
        
        return self.combine_images(lines_img,lane_img,1.0,1.0)
   
    def overlay_all(self,img,generateDebug=False):
        lines_img = self.draw_detected_lines(img,showBoxes=False,addTo=False,unwarp=False)
        lane_img = self.draw_lane(lines_img,addTo=True,unwarp=True)
        self.lane_img = np.copy(lane_img)
        lane_overlaid = self.combine_images(img,lane_img,0.75,1.0)
        info_img = self.draw_info(lane_overlaid)
        
        return info_img
        

    def feed_raw_image(self,img,debug=False,isVideo=True):
        # Undistort Camera
        undistorted = self.camera_cal.undistort(img)
        # Perspective Transform
        warped = self.persp_txfr.warp(undistorted)
        # Detect Lane Line Colors
        lc_images = []
        for ptr in self.lanecolor_detect_ptrs:
            lc_images += [ptr(warped)]
        # Feed into lane line detection
        self.feed_lc_images(lc_images,isVideo)
        
        # Overlay lane lines, lane area, and text info
        overlaid_img = self.overlay_all(undistorted)
        
        results_dict = {'undistorted':undistorted,
                'warped':warped,
                'lc_images':lc_images,
                'overlaid':overlaid_img}
    
        # Debugging
        if debug:
            debug_img = self.generate_debug_img(lc_images[0])
            results_dict['debug_img'] = debug_img
                    
        return results_dict
        
    def feed_lc_images(self,imgs,isVideo=False):
        self.left_line.feed_lc_images(imgs,warm=isVideo)
        self.right_line.feed_lc_images(imgs,warm=isVideo)
        #print("feeding fits")
        if self.left_line.detected:
            self.smart_fit.feed_left_fit(self.left_line.fit,isVideo)
        if self.right_line.detected:
            self.smart_fit.feed_right_fit(self.right_line.fit,isVideo)
        return
