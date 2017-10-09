# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:17:18 2017

@author: Alex
"""

import numpy as np
import scipy as scy

def gaussian(x, mu, std_dev):    
    dist = (x - mu) ** 2
    den = 2 * std_dev ** 2
    return np.exp(- dist / den)

def sigmoid(x):
    return 1 / (1.0 + clipped_exp(-(2.0*x)))

def clipped_exp(x):
    cx =np.clip(x, -700, 700)
    return np.exp(cx)

def computate_noise(previous_noise, delta_time, tau): 
    C1 = delta_time / tau
    C2 = 1.5
    return previous_noise + C1 * (C2 * np.random.randn(*previous_noise.shape) - previous_noise)

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def Cut_range(x, x_min, x_high):
    return np.maximum(x_min, np.minimum(1,x))

def squared_distance(x1,x2):
    return np.absolute((x1 - x2)**2)

def Distance( x1 , x2 , y1 , y2  ):
    return np.sqrt( squared_distance(x1,x2) + squared_distance(y1,y2) )

def change_range(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def screw(EP3d):    
    screw_anti_EP3d = np.zeros([3,3])  
    screw_anti_EP3d[0,1] = -EP3d[2]
    screw_anti_EP3d[0,2] = EP3d[1]
    screw_anti_EP3d[1,0] = EP3d[2]
    screw_anti_EP3d[1,2] = -EP3d[0]
    screw_anti_EP3d[2,0] = -EP3d[1]
    screw_anti_EP3d[2,1] = EP3d[0]
    return screw_anti_EP3d

def conversion2d(EP3d):
    conversion_matrix = np.zeros([2,3])
    conversion_matrix[0,1] = -1
    conversion_matrix[1,2] = 1                 
    x_orient = np.array([1,0,0])
    theta = np.linalg.norm(EP3d)
    screw_anti_EP3d = screw(EP3d)
    R = scy.linalg.expm(screw_anti_EP3d * theta)
    e1 = np.dot(R,x_orient)
    EP2d = np.dot(e1, conversion_matrix.T)
    return EP2d

def error(EP, actual_position):
    return EP - actual_position 

def computeXValue(gaussian_number, interval_lenght, X):
    # computate gaussian's average values
    for gn in xrange(gaussian_number):
        if gn == 0 :
            X[0] = 0    
        else:
            X[gn] = X[gn-1] + interval_lenght
             
def placeReward(trial):
    if trial / 2 == 0:
        reward_position = np.array([0.,0.75])
    else: 
        reward_position = np.array([0.,-0.75])
    return reward_position    
    

    








