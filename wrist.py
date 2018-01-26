# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 22:38:37 2017

@author: Alex
"""

import numpy as np
import math_utils as utils



class Wrist:
    
    def init(self, maxStep, maxTrial, maxEpoch):
        
        # kg
        self.wristRange = 1.
        
        self.rangeMin = [-90., -50., -65]
        self.rangeMax = [+90., +30, 70]
        self.rangeMin = utils.change_range(utils.polar2cart(self.rangeMin), -1., 1., 0., 1.)
        self.rangeMax = utils.change_range(utils.polar2cart(self.rangeMax), -1., 1., 0., 1.)
        
        
        
        self.DOF = 3
        self.MASS = 1.
        
        
        
        self.DELTM = 0.1
        self.TAU = 1.
        
        
        
        # 3d movements
        self.curr3DPos = np.ones(3) * 0.5
        self.prv3DPos = np.ones(3) * 0.5
        
        self.ep = np.ones(3) * 0.5
        self.prvEp = np.ones(3) * 0.5
        
        self.curr3DErr = np.zeros(3)
        self.prv3DErr = np.zeros(3)      
        self.force = np.zeros(3)
        self.angVel = np.zeros(3)
        self.curr3DAcc = np.zeros(3)
        self.curr3DVel = np.zeros(3)
        self.prv3DVel = np.zeros(3)
        
        self.posBuff = np.zeros([3,maxStep])
        self.trajectories = np.zeros([3, maxStep, maxTrial, maxEpoch])
        
    def epochReset(self):
               
        self.curr3DPos = np.ones(3) * 0.5
        self.prv3DPos = np.ones(3) * 0.5                       
        
        self.ep = np.ones(3) * 0.5
        self.prvEp = np.ones(3) * 0.5
    
        self.curr3DErr *= 0
        self.prv3DErr *= 0
        self.force *= 0
        self.curr3DAcc *= 0
        self.curr3DVel *= 0
        self.prv3DVel *= 0
        
    def trialReset(self):
        
        self.curr3DPos = np.ones(3) * 0.5
        self.prv3DPos = np.ones(3) * 0.5                      
        
        self.ep = np.ones(3) * 0.5
        self.prvEp = np.ones(3) * 0.5
            
        self.curr3DErr *= 0
        self.prv3DErr *= 0
        self.angVel *= 0
        self.force *= 0
        self.curr3DAcc *= 0
        self.curr3DVel *= 0
        self.prv3DVel *= 0
    
            
        
    def pdControl(self): 

        Kp = 4.
        Kd = 1.
        self.force = Kp * self.curr3DErr - Kd * self.curr3DVel
        
        
    def saveMov3d(self):
        self.prv3DPos = self.curr3DPos.copy() 
        self.prv3DVel = self.curr3DVel.copy()
        
        
    def move3d(self): 
        self.prv3DPos = self.curr3DPos.copy() 
        self.prv3DVel = self.curr3DVel.copy()
        self.curr3DErr = self.ep - self.prv3DPos
        self.pdControl()
        self.curr3DAcc = self.force / self.MASS
        self.curr3DVel = self.prv3DVel + self.curr3DAcc * self.DELTM 
        self.curr3DPos = self.prv3DPos + self.curr3DVel * self.DELTM 
        self.curr3DPos = utils.Cut_range(self.curr3DPos, 0. , 1.)
        
        
        