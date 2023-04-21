import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from model import *

def color_select(img):
    mask_gbr_blue = cv2.inRange(img, (130, 0, 0), (255, 120, 70))
    mask_gbr_green = cv2.inRange(img, (80, 140, 0),(180, 200, 130))

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    h, s, v = cv2.split(img_hsv)   

    mask_s_blue = cv2.inRange(s, 90, 255) & cv2.inRange(h, 99, 124)             
    mask_s_green = cv2.inRange(s, 35, 255) & cv2.inRange(h, 35, 99)

    rgbs_b = mask_gbr_blue & mask_s_blue          
    rgbs_g = mask_gbr_green & mask_s_green

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 15))
    out_b = cv2.dilate(rgbs_b, kernel, 5) 
    out_g = cv2.dilate(rgbs_g, kernel, 5)

    return out_b, out_g

def check_ratio(img, contours):
    carPlateList = []                            
    for index, contour in enumerate(contours):                                                 
        rect = cv2.minAreaRect(contour)                               
        w, h = rect[1]                                                                         
        if w < h:                                                                              
            w, h = h, w                                                                        
        scale = w/h                                                                            
        if scale > 1.5 and scale < 6:                                                              
            carPlateList.append(contour) 
    return carPlateList                

def find_squares(img, contours):
    conts = []
    squares = []
    index = 0
    for cnt in contours:
        original_cnt = cnt
        cnt_len = cv2.arcLength(cnt, True) 
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) 
        if len(cnt) >= 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            index = index + 1
            squares.append(cnt)
            conts.append(original_cnt)
    return squares, conts

def grounding_identify(img):
    # identify the grounding color of license plate is blue or green
    h, w = img.shape[0], img.shape[1]
    B, G = 0, 0
    for i in range(h):
        for j in range(w):
            B += img[i, j, 0]
            G += img[i, j, 1]
    if B > G:
        return True
    else:
        return False

def detect1(img):
    if isinstance(img, str):
        img_init = cv2.imread(img)  
    else:
        img_init = img
    img_resize = img_init
    img_resize = cv2.resize(img_init, dsize=(1000, 600), fx=0, fy=0)  
    img_blur = cv2.blur(img_resize, (2, 2))  
    img_limis = color_select(img_blur)
    cont = []
    area = 0
    color = ''
    for img_limi in img_limis:
        conts = cv2.findContours(img_limi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] 
        conts = check_ratio(img_limi, contours=conts)
        area_new = sum([cv2.contourArea(c) for c in conts])
        if area_new < 100000:
            conts = find_squares(img_resize, conts)[1]
        area_new = sum([cv2.contourArea(c) for c in conts])
        if area_new > area:
            area = area_new
            cont = conts

    c = max(cont, key=cv2.contourArea) 
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect) 
    # cv2.drawContours(img_resize, [np.int0(box)], -1, (0, 0, 255), 3)  
        
    img_final = img_resize[int(min(box[1][1], box[2][1])):int(max(box[0][1], box[3][1])), 
                            int(min(box[0][0], box[1][0]))+5:int(max(box[2][0], box[3][0]))-5]
    avg_b = np.mean(img_final.transpose((2,0,1))[0])
    avg_g = np.mean(img_final.transpose((2,0,1))[1])

    if avg_b > avg_g:
        color = 'blue'
        img_final_resize = cv2.resize(img_final, dsize=(220, 70), fx=0, fy=0)
    else:
        color = 'green'
        img_final_resize = cv2.resize(img_final, dsize=(240, 70), fx=0, fy=0)

    return img_final_resize, color

def find_waves(threshold, histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks

def segment(color, card_img):
    chars = []
    flag = 0
    if color == 'blue':
        gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        x_histogram = np.sum(gray_img, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2
        wave_peaks = find_waves(x_threshold, x_histogram)

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        gray_img = gray_img[wave[0]:wave[1]]
        row_num, col_num = gray_img.shape[:2]

        gray_img = gray_img[2: row_num - 2]
        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        y_threshold = (y_min + y_average) / 6

        wave_peaks = find_waves(y_threshold, y_histogram)
        
        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]

        if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] < 3:
            wave_peaks.pop(0)
        if wave_peaks[-1][1] - wave_peaks[-1][0] < max_wave_dis / 3:
            wave_peaks.pop(-1)

        point = wave_peaks[2]
        if point[1] - point[0] < max_wave_dis / 3:
            point_img = gray_img[:, point[0]:point[1]]
            if np.mean(point_img) < 255 / 4:
                wave_peaks.pop(2)

        for wave in wave_peaks:
            resize = (13, 30)
            seg = cv2.resize(gray_img[:, wave[0]: wave[1]], resize)
            chars.append(seg)
        if len(chars) > 7:
            flag = 1
    
    if color == 'green' or flag:
        gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        if not flag:
            gray_img = cv2.bitwise_not(gray_img)
        gray_img = gray_img[2: gray_img.shape[:2][0] - 2]
        ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 9))
        dilate_img = cv2.dilate(gray_img, kernel, 4) 

        conts, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chars, segs = [], []
        for c in conts:
            x, y, w, h = cv2.boundingRect(c)
            s = w * h
            if s < 20000:
                segs.append((x, w, s))
                # cv2.rectangle(card_img, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.imwrite('imgs/2-19.png', card_img)

        if not flag:
            resize = (10, 30)
            segs = sorted(sorted(segs, key=lambda x: x[2], reverse=True)[:8], key=lambda x: x[0])
        else:
            resize = (12, 30)
            segs = sorted(sorted(segs, key=lambda x: x[2], reverse=True)[:7], key=lambda x: x[0])
        for i, s in enumerate(segs):
            if i == 0:
                seg = cv2.resize(gray_img[:, 5: s[1]+s[0]], resize)
            else:
                seg = cv2.resize(gray_img[:, s[0]: s[1]+s[0]], resize)
            chars.append(seg)

        # plt.figure(figsize=(6,2))
        # for i in range(len(chars)):
        #     plt.subplot(1, 8, i+1)
        #     plt.imshow(chars[i], cmap='gray')
        # plt.show()
    return chars

def rectify(img):
    # hsv and Gaussian blur
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_blur = cv2.GaussianBlur(img_hsv, [5, 5], 0)

    # hvs color segmentation to determine car licence plate area according to its blue color
    img_mask = cv2.inRange(img_hsv_blur, np.array([100, 115, 115]), np.array([124, 255, 255]))
    #cv2.imshow('img_mask_blue', img_mask)

    # morphology open to eliminate noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)

    # morphology close to get whole licence area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 2)
    #cv2.imshow('img_lcs', img_lcs)

    # find contours of areas which possibly are license plate
    contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # if the license plate is of green grounding, repeat the above process
    if len(contours) == 0:
        img_mask = cv2.inRange(img_hsv_blur, np.array([35, 10, 160]), np.array([70, 100, 200]))
        #cv2.imshow('img_mask_green', img_mask)

        # morphology open to eliminate noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations = 1)

        # morphology close to get whole licence area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations = 1)
        #cv2.imshow('img_lcs_green', img_lcs)

        # find contours of areas which possibly are license plate 
        contours, hierarchy = cv2.findContours(img_lcs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cut the oblique licence plate out and get the oblique angle
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w > h * 1.2 and w < h * 2.2 and w > 410:
            lcs_oblique = img[y:y + h - 25, x:x + w - 5]
            lcs_oblique_bin = img_lcs[y:y + h - 25, x:x + w - 5]
            angle = cv2.minAreaRect(i)[2]
            break
    
    # compute the vertex of the license plate
    lcs_area = np.where(lcs_oblique_bin==255)
    x1, y1, x2, y2 = min(lcs_area[1]), min(lcs_area[0]), max(lcs_area[1]), max(lcs_area[0])
    dx = x2 - x1
    dy = lcs_area[0].shape[0] // dx
    
    if angle < 45:
        src_vertex = np.array([[x1, y1], [x2, y2 - dy], [x1, y1 + dy], [x2, y2]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1 + int(1.5 * dx), y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    elif angle < 70:
        src_vertex = np.array([[x1, y2 - dy + 25], [x1, y2], [x2, y1 + 20], [x2 - 5, y1 + dy - 30]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)
    else:
        src_vertex = np.array([[x1 + 20, y2 - dy + 27], [x1, y2], [x2, y1 + 20], [x2 - 25, y1 + dy - 20]], dtype=np.float32)
        dst_vertex = np.array([[x1, y1], [x1, y1 + dy], [x1 + int(1.5 * dx), y1], [x1 + int(1.5 * dx), y1 + dy]], dtype=np.float32)

    # perspective transform to get the orthogonal license plate
    M = cv2.getPerspectiveTransform(src_vertex, dst_vertex)
    lcs_orth = cv2.warpPerspective(lcs_oblique, M, (int(1.5 * dx), dy))
    return lcs_orth

def detect2(img):
    # resize
    img_bgr = cv2.resize(img, (int(200*img.shape[1]/img.shape[0]), 200))
    # identify the grounding color is blue or green
    isBlue = grounding_identify(img_bgr)

    # gray
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to eliminate noise
    img_blur = cv2.GaussianBlur(img_gray, [5, 5], 5)

    # threshold to get binary image to emphasize characters
    if isBlue == False:
        ret, img_thresh = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
    # morphology open
    if isBlue == False:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # dilate every character for easier split 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 50))
    img_dilated = cv2.dilate(img_open, kernel, iterations=1)
    # cv2.imshow('img_dilated', img_dilated)

    color = 'blue' if isBlue else 'green'

    return img_open, img_dilated, color

def detectAndSegment(img, level):
    if level == 2: # for difficult level
        img = rectify(img)
        _, _, color = detect2(img)
        imgs = segment(color, img)
    else:
        img, color = detect1(img)
        img, color = detect1(img) # pre-process twice
        imgs = segment(color, img)
    return imgs, color, img

img = cv2.imread(r'resources\images\easy\1-2.jpg')
detectAndSegment(img, 0)