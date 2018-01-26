# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 22:30:39 2017

@author: Alex
"""

import numpy as np
import scipy as scy
#☺import robotics as rob
#import scipy.spatial.distance as ssd


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
    return 1 / (1.0 + clipped_exp(-(2.0*x)))




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




def conversion2d(curr3DPos, perturbation, pertMag):
    
 #   curr3DPosSP = cart2sph(curr3DPos)
  #  pertMag = 180 - pertMag
    
 #   print curr3DPos
    conversion_matrix = np.zeros([2,3])
    conversion_matrix[0,1] = -1.
    conversion_matrix[1,2] = 1.    
                     
    ex = np.array([1.,0.,0.])
#•    exHat = screwMatrix(ex)
    
   # ey = np.array([0.,1.,0.])
  #  eyHat = screwMatrix(ey)
    
   # ez = np.array([0.,0.,1.])
  #  ezHat = screwMatrix(ez)
    antiSimMatrix = screwMatrix(curr3DPos)
  #  R = rotation(curr3DPos)
 #   print R
    
   # normV = rob.Normalize(curr3DPos)
 #   r = np.array([-exHat * curr3DPos[0], eyHat * curr3DPos[1],ezHat * curr3DPos[2]])
  #  expC3 = scy.linalg.expm((rob.VecToso3(curr3DPos)))
  #  axis, theta = rob.AxisAng3(expC3)
    
 #   theta = scy.linalg.norm(expC3)
 #   r = curr3DPos * theta
 #   rHat = rob.VecToso3(r)
    R = scy.linalg.expm(antiSimMatrix)
    
    
  #  if perturbation == False:
  #  R = np.dot(np.dot(scy.linalg.expm(-exHat * curr3DPos[0]), scy.linalg.expm(eyHat * curr3DPos[1])),scy.linalg.expm(ezHat * curr3DPos[2]))
 #   print np.linalg.det(R)
   # else:
   #     R = (np.dot(np.dot(scy.linalg.expm(-exHat * curr3DPos[0]), scy.linalg.expm(ezHat * curr3DPos[2])),scy.linalg.expm(eyHat * curr3DPos[1]))).T
#    e1 = np.dot(R,ex)
    EP2d = np.dot(np.dot(conversion_matrix, R), ex)
   # print EP2d
    if perturbation == True:
        pertMag = np.deg2rad(pertMag)
        rotEP2d = np.zeros(2)
    #    rotEP2d = perturbation(EP2d, pertMag)
#        print rotEP2d
        rotEP2d[0] = EP2d[0] * np.cos(pertMag) - EP2d[1] * np.sin(pertMag)
        rotEP2d[1] = EP2d[0] * np.sin(pertMag) + EP2d[1] * np.cos(pertMag)
      #  EP2d = rotEP2d.copy() 
        return rotEP2d
    else:
        #print EP2d
        return EP2d




    
def perturbation(EP2d, pertMag):
    
    rotMat = np.zeros([2,2])
    rotMat[0,0] = np.cos(pertMag)
    rotMat[0,1] = -np.sin(pertMag)
    rotMat[1,0] = np.sin(pertMag)
    rotMat[1,1] = np.cos(pertMag)
    
    return np.dot(EP2d,rotMat)



    
    

def build3DGrid(intervals, rAnge):
    x = np.linspace(0,rAnge ,intervals)
    y = np.linspace(0,rAnge,intervals)
    z = np.linspace(0,rAnge,intervals)

    xxx, yyy, zzz = np.meshgrid(x,y,z)

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



def polar2cart(polar):
    x = np.zeros(3)
    x[0] = np.sin(polar[2]) * np.cos(polar[1])
    x[1] = np.sin(polar[2]) * np.sin(polar[1])
    x[2] = np.cos(polar[2])
    return x

def rotation(theta):
    thetaD = np.rad2deg(theta)
    Rx = np.array([[1,0,0], [0, np.cos(thetaD[0]), -np.sin(thetaD[0])], [0, np.sin(thetaD[0]), np.cos(thetaD[0])]])
    Ry = np.array([[np.cos(thetaD[1]), 0, -np.sin(thetaD[1])], [0, 1, 0], [np.sin(thetaD[1]), 0, np.cos(thetaD[1])]]) 
    Rz = np.array([[np.cos(thetaD[2]), -np.sin(thetaD[2]), 0], [np.sin(thetaD[2]), np.cos(thetaD[2]), 0], [0,0,1]]) 
  #  R = np.dot(Rx, np.dot(Ry, Rz))
   # print R
    return np.dot(Rx, np.dot(Ry, Rz))       

def make_axis_rotation_matrix(direction, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)

     Parameters:

         angle : float a
         direction : array d
     """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    eye = np.eye(3, dtype=np.float64)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                     [-d[2],     0,  d[0]],
                     [d[1], -d[0],    0]], dtype=np.float64)

    mtx = ddt + np.cos(angle) * (eye - ddt) + np.sin(angle) * skew
  #  print mtx
    return mtx