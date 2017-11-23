# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:17:18 2017

@author: Alex
"""

import numpy as np
import scipy as scy
#import scipy.spatial.distance as ssd

#a =(1. / (2 * np.pi* sig**2)) 
#b =(1. / ((np.sqrt((2 * np.pi))**3)* sig**3))
def gaussian(x, mu, sig):
    return np.exp(-(x - mu)**2/ (2 * sig ** 2.))

def gaussian2D(curr2DPos, mu, sig, inputArray, intervals):
    for j in xrange(intervals):
        for i in xrange(intervals):
            inputArray[i,j] =  np.exp(- np.sum(squared_distance(curr2DPos,mu[:,i,j])) / (2 * sig ** 2.))       
    return inputArray

def gaussian3D(curr3DPos, mu, sig, wristRawPosState, interN):
    
    for k in xrange(interN):
        for j in xrange(interN):
            for i in xrange(interN):
                wristRawPosState[i,j,k] = np.exp(- np.sum(squared_distance(curr3DPos,mu[:,i,j,k]))  / (2 * sig ** 2.))

    return wristRawPosState
    
def sigmoid(x):
    return 1 / (1.0 + np.exp(-(2.0*x)))

def clipped_exp(x):
    cx =np.clip(x, -700, 700)
    return np.exp(cx)

def computate_noise(previous_noise, delta_time, tau): 
    C1 = delta_time / tau
    C2 = 1.
    return previous_noise + C1 * (C2 * np.random.randn(*previous_noise.shape) - previous_noise)

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def Cut_range(x, x_min, x_high):
    return np.maximum(x_min, np.minimum(x_high,x))

def squared_distance(x1,x2):
    return (x1 - x2)**2

def distance2D(a , b):
    return np.sqrt(np.sum(squared_distance(a,b)))

def distance3D(a, b):
    return np.sqrt(np.sum(squared_distance(a[0],b[0]),squared_distance(a[1],b[1]), squared_distance(a[2],b[2] )))

def change_range(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def screwMatrix(EP3d):    
    antiSimMatrix = np.zeros([3,3])  
    antiSimMatrix[0,1] = -EP3d[2]
    antiSimMatrix[0,2] = EP3d[1]
    antiSimMatrix[1,0] = EP3d[2]
    antiSimMatrix[1,2] = -EP3d[0]
    antiSimMatrix[2,0] = -EP3d[1]
    antiSimMatrix[2,1] = EP3d[0]
    return antiSimMatrix

def conversion2d(EP3d):
    conversion_matrix = np.zeros([2,3])
    conversion_matrix[0,1] = -1
    conversion_matrix[1,2] = 1                 
    x_orient = np.array([1,0,0])
    theta = np.linalg.norm(EP3d)
    antiSimMatrix = screwMatrix(EP3d)
    R = scy.linalg.expm(antiSimMatrix * theta)
    e1 = np.dot(R,x_orient)
    EP2d = np.dot(e1, conversion_matrix.T)
    return EP2d

def build3DGrid(intervals, rAnge):
    x = np.linspace(0,rAnge ,intervals)
    y = np.linspace(0,rAnge,intervals)
    z = np.linspace(0,rAnge,intervals)

    xxx, yyy, zzz = np.meshgrid(x, y,z)

    end = np.zeros([3,intervals,intervals,intervals])    
    end[0,:,:,:] = xxx
    end[1,:,:,:] = yyy
    end[2,:,:,:] = zzz
        
    return end

def build2DGrid(intervals, rAnge):
    x = np.linspace(0,rAnge ,intervals)
    y = np.linspace(0,rAnge,intervals)
    xx, yy = np.meshgrid(x, y)
    
    end = np.zeros([2,intervals,intervals])  
    end[0,:,:] = xx
    end[1,:,:] = yy
        
    return end

def polar2cart(r, polar):
    x = np.zeros(3)
    x[0] = np.sin(polar[2]) * np.cos(polar[1])
    x[1] = np.sin(polar[2]) * np.sin(polar[1])
    x[2] = np.cos(polar[2])
    return x
             

