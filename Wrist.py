# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:25:29 2017

@author: Alex
"""

import numpy as np
import math_utils as utils
import cmath


class Wrist:
    
    def init(self):
        
        # kg
        self.wristRange = 1.
        self.DOF = 3
        self.MASS = 1.
        
        
        
        self.DELTM = 0.1
        self.TAU = 1.
        
        
        
        # 3d movements
        self.curr3DPos = np.ones(3) * 0.5
        self.prv3DPos = np.ones(3) * 0.5
        
        # noise
        self.currNoise = np.zeros(self.DOF)
        self.prvNoise = np.zeros(self.DOF)
        
        
        
        
        
        
        
        
        
  
         
        
        
        self.ep = np.ones(3) * 0.5
        self.prvEp = np.ones(3) * 0.5
        
        
        self.force = np.zeros(3)
        self.curr3DErr = np.zeros(3)
        self.curr3DVel = np.zeros(3)
        self.curr3DAcc = np.zeros(3)
        self.prv3DErr = np.zeros(3)
        self.prv3DPos =np.zeros(3)
        self.prv3DVel = np.zeros(3)
        
        
    
        
        
        
    def pdControl(self): 
        Kp = 0.75
        Kd = 0.25
        self.force = Kp * (self.curr3DErr) + Kd * utils.derivative(self.prv3DErr, self.curr3DErr, self.DELTM, self.TAU)
        
        
    def saveMov3d(self):
        self.prv3DErr = self.curr3DErr.copy()
        self.prv3DPos = self.curr3DPos.copy() 
        self.prv3DVel = self.curr3DVel.copy()
        
        
    def move3d(self): 
        self.curr3DErr = (self.ep - self.prv3DPos).copy()
        self.pdControl()
        self.curr3DAcc = (self.force / self.MASS).copy()
        self.curr3DVel = (self.prv3DVel + self.curr3DAcc * self.DELTM).copy()
        self.curr3DPos = (self.prv3DPos + self.curr3DVel * self.DELTM).copy() 
        self.curr3DPos[0] = utils.Cut_range(self.curr3DPos[0], 0.1 , 0.9)
        self.curr3DPos[1] = utils.Cut_range(self.curr3DPos[1], 0.1 , 0.9)
        self.curr3DPos[2] = utils.Cut_range(self.curr3DPos[2], 0.1 , 0.9)
        
      #  self.curr3DPos = np.(self.curr3DPos)
       # self.curr3DPos = (utils.polar2cart(0.5, self.curr3DPos)).copy()
     #   self.curr3DPos[1] = utils.Cut_range(self.curr3DPos[1],np.sin(np.deg2rad(-30))+ 0.5, np.sin(np.deg2rad(+30))).copy()
     #   self.curr3DPos[2] = utils.Cut_range(self.curr3DPos[2], 0.3, 0.9).copy()
 
    
    
    
    
  
    
    
    
    

                
                