from os import makedirs
from os.path import exists, basename
from shutil import copyfile
import matplotlib
from PIL import Image
import math
from skimage.measure import label, regionprops, regionprops_table
from predictor import Predictor

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

import keras
from properties import IMG_SIZE
import xml.etree.ElementTree as ET
import potrace
import numpy as np
import re

import vectorization_tools
import rasterization_tools

NAMESPACE = '{http://www.w3.org/2000/svg}'

def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0

    return x_reshape

def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)
      
# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v

def print_image(filename, image, cmap=''):
    if cmap != '':
        plt.imsave(filename, image.reshape(28, 28), cmap=cmap)
    else:
        plt.imsave(filename, image.reshape(28, 28))

def bitmap_count(digit, threshold):    
    bw = np.asarray(digit.purified).copy()    
    #bw = bw / 255.0    
    count = 0    
    for x in np.nditer(bw):
        if x > threshold:
            count += 1    
    return count

def controlpoint_count(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('[CL]')
    segments = pattern.findall(svg_path)    
    num_matches = len(segments)
    return num_matches

def move_distance(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sM\s([\d\.]+),([\d\.]+)')
    segments = pattern.findall(svg_path) 
    if len(segments) > 0:
        dists = [] # distances of moves
        for segment in segments:     
            x1 = float(segment[0])
            y1 = float(segment[1])
            x2 = float(segment[2])
            y2 = float(segment[3])
            dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
            dists.append(dist)
        return int(np.sum(dists))
    else:
        return 0

def orientation_calc(digit, threshold):
    x = []
    y = []
    bw = np.asarray(digit.purified).copy()  
    for iz,ix,iy,ig in np.ndindex(bw.shape):
        if bw[iz,ix,iy,ig] > threshold:
            x.append([iy])          
            y.append(ix)
    X = np.array(x)
    Y = np.array(y)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)
    normalized_ori = (-lr.coef_ + 2)/4
    # scale to be between 0 and 100
    new_ori = normalized_ori * 100
    return int(new_ori)       

def rescale(solutions, perfs, new_min = 0, new_max = 24):
    max_shape = new_max + 1
    output1 = np.full((max_shape,max_shape), None,dtype=(object))
    output2 = np.full((max_shape,max_shape), 2.0, dtype=(float))
    
        
    old_min_i = 0
    old_min_j = 0
    old_max_i, old_max_j = solutions.shape[0], solutions.shape[1]

    for (i,j), value in np.ndenumerate(perfs):
        new_i = int(((new_max - new_min) / (old_max_i - old_min_i)) * (i - old_min_i) + new_min)
        new_j = int(((new_max - new_min) / (old_max_j - old_min_j)) * (j - old_min_j) + new_min)
        if value != 2.0:
            if output2[new_i, new_j] == 2.0 or value < output2[new_i,new_j]:
                output2[new_i,new_j] = value
                output1[new_i,new_j] = solutions[i,j]
    return output1, output2