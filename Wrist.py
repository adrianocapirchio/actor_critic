# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:25:29 2017

@author: Alex
"""

import numpy as np
import math_utils as utils


class Wrist:
    
    def init(self):
        
        # kg
        self.MASS = 1.
        self.DELTATM = 0.1
        self.TAU = 1.
        
        self.agent_starting_3dposition = np.zeros(3)
        self.agent_starting_2dposition = np.zeros(2)
        
        # 3d movements
        self.force = np.zeros(3)
        self.actual_3derror = np.zeros(3)
        self.previous_3derror = np.zeros(3)
        self.actual_3dposition = np.zeros(3)
        self.previous_3dposition = np.zeros(3)
        self.actual_3dvelocity = np.zeros(3)
        self.previous_3dvelocity = np.zeros(3)
        self.actual_3dacceleration = np.zeros(3)
        
        # 2d movements
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
        
    def saveMov3d(self, actual_3derror, actual_3dposition, actual_3dvelocity, actual_3dacceleration):
        previous_3derror = actual_3derror.copy()
        previous_3dposition = actual_3dposition.copy() 
        previous_3dvelocity = actual_3dvelocity.copy()
        previous_3dacceleration = actual_3dacceleration.copy()
        return previous_3derror, previous_3dposition, previous_3dvelocity, previous_3dacceleration
        
    def move3d(self, ep3d): 
        self.actual_3derror = utils.error(ep3d, self.actual_3dposition)
        self.force = self.pdControl()
        self.actual_3dacceleration = self.force / self.MASS
        self.actual_3dvelocity = self.previous_3dvelocity + self.actual_3dacceleration * self.DELTATM
        self.actual_3dposition = self.previous_3dposition + self.actual_3dvelocity * self.DELTATM
        self.actual_3dposition = utils.Cut_range(self.actual_3dposition, 0.00001, 0.99999)
        return self.actual_3dposition  
    
    def saveMov2d(self, actual_2derror, actual_2dposition, actual_2dvelocity, actual_2dacceleration):
        previous_2derror = actual_2derror.copy()
        previous_2dposition = actual_2dposition.copy() 
        previous_2dvelocity = actual_2dvelocity.copy()
        previous_2dacceleration = actual_2dacceleration.copy() 
        return previous_2derror, previous_2dposition, previous_2dvelocity, previous_2dacceleration
    
    
        

                
                