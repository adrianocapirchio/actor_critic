# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:07:27 2017

@author: Alex
"""

import numpy as np
import math_utils as utils

class Cerebellum():
    
    def init(self, multiNet, stateBg, DOF = 3):
        
        self.DELTM = 0.1
        self.TAU = 1.
        
        self.cbETA = 0.01
        
        self.actState = np.zeros(len(stateBg))
        
        
        self.w = np.zeros([len(self.actState), DOF])
        
        if multiNet == True:
            self.w0 = self.w.copy()
            self.w1 = self.w.copy()
            self.w2 = self.w.copy()
            self.w3 = self.w.copy()
            self.w4 = self.w.copy()
            self.w5 = self.w.copy()
            self.w6 = self.w.copy()
            self.w7 = self.w.copy()
            self.w8 = self.w.copy()   
            
        self.currNoise = np.zeros(DOF)
        self.prvNoise = np.zeros(DOF)
        
        
        self.currOut = np.zeros(DOF)
        self.prvOut = np.zeros(DOF)
        self.trainOut = np.zeros(DOF)
        self.error = np.zeros(DOF)
        
    def computate_noise(self): 
        self.C1 = self.DELTM/ self.TAU
        self.C2 = 0.00
        self.currNoise = self.prvNoise + self.C1 * (self.C2 * np.random.randn(*self.prvNoise.shape) - self.prvNoise)
      #  self.currNoise = utils.change_range(self.currNoise, -1., 1., -0.5, 0.5)
                                            
    def trainCb(self,state, ep):
        self.trainOut[0] = utils.sigmoid(np.dot(self.w[:,0], state)).copy()
        self.trainOut[1] = utils.sigmoid(np.dot(self.w[:,1], state)).copy()
        self.trainOut[2] = utils.sigmoid(np.dot(self.w[:,2], state)).copy()
        self.error = (ep - self.trainOut).copy()
        self.w += self.cbETA * np.outer(state, self.error) 
       # (np.sum(surp))/len(np.trim_zeros(surp)) *
   #     print self.w                                       
    def spreading(self,state):
        self.currOut[0] = utils.sigmoid(np.dot(self.w[:,0], state)).copy()
        self.currOut[1] = utils.sigmoid(np.dot(self.w[:,1], state)).copy()
        self.currOut[2] = utils.sigmoid(np.dot(self.w[:,2], state)).copy()
        
    
    