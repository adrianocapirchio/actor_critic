# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:25:29 2017

@author: Alex
"""

import numpy as np
import math_utils as utils


class Wrist:
    
    def init(self, gaussian_number):
        
        # kg
        self.MASS = 1.
        self.DELTATM = 0.1
        self.TAU = 1.
        
        # 3d movements
        self.position3d_state = np.zeros(gaussian_number*3)
        self.force = np.zeros(3)
        self.actual_3derror = np.zeros(3)
        self.previous_3derror = np.zeros(3)
        self.actual_3dposition = np.zeros(3)
        self.previous_3dposition = np.zeros(3)
        self.actual_3dvelocity = np.zeros(3)
        self.previous_3dvelocity = np.zeros(3)
        self.actual_3dacceleration = np.zeros(3)
        
        # 2d movements
        self.reward_state = np.zeros(gaussian_number*2)
        self.position2d_state = np.zeros(gaussian_number*2)
        self.actual_2derror = np.zeros(2)
        self.previous_2derror = np.zeros(2)
        self.next_2dposition = np.zeros(2)
        self.actual_2dposition = np.zeros(2)
        self.previous_2dposition = np.zeros(2)
        self.actual_2dvelocity = np.zeros(2)
        self.previous_2dvelocity = np.zeros(2)
        self.actual_2dacceleration = np.zeros(2)
        
    def pdControl(self): 
        Kp = 21.
        Kd = 8.
        force = Kp * (self.actual_3derror) + Kd * utils.derivative(self.actual_3derror, self.previous_3derror, self.DELTATM, self.TAU)
        return force
        
    def saveMov3d(self):
        self.previous_3derror = self.actual_3derror.copy()
        self.previous_3dposition = self.actual_3dposition.copy() 
        self.previous_3dvelocity = self.actual_3dvelocity.copy()
        self.previous_3dacceleration = self.actual_3dacceleration.copy()
        
        
    def move3d(self, ep3d): 
        self.actual_3derror = utils.error(ep3d, self.actual_3dposition)
        self.force = self.pdControl()
        self.actual_3dacceleration = self.force / self.MASS
        self.actual_3dvelocity = self.previous_3dvelocity + self.actual_3dacceleration * self.DELTATM
        self.actual_3dposition = self.previous_3dposition + self.actual_3dvelocity * self.DELTATM
        self.actual_3dposition = utils.Cut_range(self.actual_3dposition, 0.00001, 0.99999)
        return self.actual_3dposition  
    
    
    
    

                
                