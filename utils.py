
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from os import makedirs
from os.path import exists, basename
from shutil import copyfile
import matplotlib
from PIL import Image
import math
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
import csv
import json
import glob
# For Python 3.6 we use the base keras
import keras

import xml.etree.ElementTree as ET
import potrace
import numpy as np
import re
import copy
# local imports

import vectorization_tools
import rasterization_tools
from properties import IMG_SIZE, INTERVAL

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
    np.save(filename, image)

def bitmap_count(digit, threshold): 
    image = copy.deepcopy(digit.purified)   
    bw = np.asarray(image) 
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
    image = copy.deepcopy(digit.purified)   
    bw = np.asarray(image)   
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
    output2 = np.full((max_shape,max_shape), np.inf, dtype=(float))
    
        
    old_min_i = 0
    old_min_j = 0
    old_max_i, old_max_j = solutions.shape[0], solutions.shape[1]

    for (i,j), value in np.ndenumerate(perfs):
        new_i = int(((new_max - new_min) / (old_max_i - old_min_i)) * (i - old_min_i) + new_min)
        new_j = int(((new_max - new_min) / (old_max_j - old_min_j)) * (j - old_min_j) + new_min)
        if value != np.inf:
            if output2[new_i, new_j] == np.inf or value < output2[new_i,new_j]:
                output2[new_i,new_j] = value
                output1[new_i,new_j] = solutions[i,j]
    return output1, output2

def new_rescale(features, perfs, new_min_1, new_max_1, new_min_2, new_max_2):
    shape_1 = new_max_1 - new_min_1 + 1
    shape_2 = new_max_2 - new_min_2 + 1
    output2 = np.full((shape_2, shape_1), np.inf, dtype=(float))

    old_min_i = 0
    old_min_j = 0
    old_max_i = perfs.shape[0]
    old_max_j = perfs.shape[1]

    for (i, j), value in np.ndenumerate(perfs):
        new_j = int(((new_max_1 - new_min_1) / (old_max_j - old_min_j)) * j - new_min_1)
        new_i = int(((new_max_2 - new_min_2) / (old_max_i - old_min_i)) * i - new_min_2)
        if value != np.inf:
            if output2[new_i, new_j] == np.inf or value < output2[new_i, new_j]:
                output2[new_i, new_j] = value
                #output1[new_i, new_j] = solutions[i, j]
    return output2

def generate_reports(filename, log_dir_path): 
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    # write the header
    cf.writerow(["Features","Time","Covered seeds","Filled cells","Filled density", "Misclassified seeds","Misclassification","Misclassification density"])
        
    jsons = [f for f in sorted(glob.glob(f"{log_dir_path}/*.json"),  key=os.path.getmtime) if "Bitmaps_Moves" in f]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Bitmaps,Moves",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)

    jsons = [g for g in sorted(glob.glob(f"{log_dir_path}/*.json"),  key=os.path.getmtime) if "Moves_Orientation" in g]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Orientation,Moves",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)

    jsons = [h for h in sorted(glob.glob(f"{log_dir_path}/*.json"), key=os.path.getmtime) if "Bitmaps_Orientation" in h]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Bitmaps,Orientation",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)
