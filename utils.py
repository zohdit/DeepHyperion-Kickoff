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
from properties import IMG_SIZE, RESULTS_PATH
import xml.etree.ElementTree as ET
import potrace
import numpy as np
import re

from curvature import findCircle
from DSA_calculator import DSA_calculator
import vectorization_tools
import rasterization_tools

NAMESPACE = '{http://www.w3.org/2000/svg}'
# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


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


def print_archive(archive):
    path = RESULTS_PATH + '/archive'
    dst = path + '/'
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename = dst + basename(
            'archived_' + str(i) + '_label_' + str(ind.member.predicted_label) + '_seed_' + str(ind.seed))
        plt.imsave(filename, ind.member.purified.reshape(28, 28), cmap=cm.gray)
        np.save(filename, ind.member.purified)
        assert (np.array_equal(ind.member.purified, np.load(filename + '.npy')))

def print_misclass_archive(archive):
    path = RESULTS_PATH + '/misclassarchive'
    dst = path + '/'
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename = dst + basename(
            'archived_' + str(i) + '_label_' + str(ind.member.predicted_label) + '_seed_' + str(ind.seed))
        plt.imsave(filename, ind.member.purified.reshape(28, 28), cmap=cm.gray)
        #loaded_label = (Predictor.predict(ind.member.purified))
        #assert (ind.member.predicted_label == loaded_label[0])
        #assert (ind.member.predicted_label == Predictor.randomized_model.predict_classes(ind.member1.purified))
        np.save(filename, ind.member.purified)
        #loaded_label = Predictor.predict(np.load(filename + '.npy'))[0]
        #assert (ind.member.predicted_label == loaded_label)
        assert (np.array_equal(ind.member.purified, np.load(filename + '.npy')))
        
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

def print_image(filename, image, cmap):
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

def move_count(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('[M]')
    segments = pattern.findall(svg_path)    
    num_matches = len(segments)
    if num_matches > 0:
        return num_matches-1
    else:
        return 0  

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

def angle_calc(digit):
    angles = []
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sC\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    for segment in segments:        
        angle = curv_angle(float(segment[0]), float(segment[1]), float(segment[2]), float(segment[3]), float(segment[4]), float(segment[5]), float(segment[6]), float(segment[7]))
        angles.append(angle)
    avg_ang = np.min(angles)
    return int(avg_ang)

def curv_angle(x11, y11, x12, y12, x21, y21, x22, y22):
    sameside = False
    if x22 - x11 == 0:
        x = x11
        if (x > x12 and x > x21) or (x < x12 and x < x21):
            sameside = True #both on same side
    else:
        m = (y22 - y11) / (x22 - x11)
        b = y11 - m * x11
        if (y12 > m * x12 + b and y21 > m * x21 + b) or (y12 < m * x12 + b and y21 < m *x21 + b):
            sameside = True #both on same side
    
    if sameside == True:
        if x12 - x11 == 0:
            A = 90
        else:
            y = (y12 - y11)
            x = (x12 - x11)  
            # A = angle between x-axis and line 1
            A = math.atan2(x,y) * 180 / math.pi
            A = np.abs((A + 180) % 360 - 180)

        if x22 - x21 == 0:
            B = 90
        else:
            y = (y22 - y21)
            x = (x22 - x21)  
            # B = angle between x-axis and line 2
            B = math.atan2(x, y) * 180 / math.pi
            B = np.abs((B + 180) % 360 - 180)
        #Angle between line 1 and line 2 = A - B
        angle = np.abs(A - B)
        return angle
    else:        
        # first angle

        if x12 - x11 == 0:
            A = 90
        else:
            y = (y12 - y11)
            x = (x12 - x11)  
            A = math.atan2(x,y) * 180 / math.pi
            A = np.abs((A + 180) % 360 - 180)

        if x21 - x12 == 0:
            B = 90
        else:
            y = (y21 - y12)
            x = (x21 - x12)         
            B = math.atan2(x,y) * 180 / math.pi
            B = np.abs((B + 180) % 360 - 180)

        #Angle between line 1 and line 2 = A - B
        angle1 = np.abs(A - B)

        # second angle

        if x21 - x12 == 0:
            A = m90
        else:
            y = (y21 - y12)
            x = (x21 - x12)       
            A = math.atan2(x,y) * 180 / math.pi
            A = np.abs((A + 180) % 360 - 180)
        if x22 - x21 == 0:
            B = 90
        else:
            y = (y22 - y21)
            x = (x22 - x21)        
            B = math.atan2(x,y) * 180 / math.pi        
            B = np.abs((B + 180) % 360 - 180)
        #Angle between line 1 and line 2 = A - B
        angle2 = np.abs(A - B)

        return np.min([angle1, angle2])

def orientation_calc(digit):
    image = digit.purified
    img = image.reshape(28, 28)
    label_img = label(img)
    props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'filled_area'
                                                 ))
    if len(props['orientation']) > 0:
        orientation = props['orientation'][0]  
    else:
        orientation = 0
    # orientation is between -pi/2 and pi/2
    # normalization
    # range = max(a) - min(a)
    # a = (a - min(a)) / range
    normalized_ori = (orientation + math.pi/2)/math.pi
    # scale to be between 0 and 100
    new_ori = normalized_ori * 100
    return int(new_ori)

def new_orientation_calc(digit, threshold):
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

def DSA_calc(digit, label, seed):
    correctly_classified, ff, predicted_label = DSA_calculator.predict(img=digit.purified,
                                                                        label=label, 
                                                                        seed=seed)
    return int(ff)

def min_radius_calc(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)
    road_list = []
    for segment in segments:
        road_list.append([float(segment[0]),float(segment[1])])
        road_list.append([float(segment[2]),float(segment[3])])
        road_list.append([float(segment[4]),float(segment[5])])
    curvatures = []
    for j in range(3):
        road = np.array(road_list)[:, :2][int(j)::3 + int(j)]        
        res = list(map(list, zip(list(road), list(road[1:]), list(road[2:]))))
        for i in range(len(res)):
            joined_lists = [*res[i][0], *res[i][1], *res[i][2]]
            curvatures.append(
                findCircle(joined_lists[0], joined_lists[1], joined_lists[2], joined_lists[3], joined_lists[4],
                            joined_lists[5]))
    radius = np.min(curvatures)*3.280839895
    return int(radius)

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