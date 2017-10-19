# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:36:46 2017

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:36:46 2017
@author: Alex
"""

import numpy as np

class ActorCritic:
    
    def init(self, DOF = 3):
        
        # GENERAL PARAMETERS
        self.interN = 29 # define intervals used to define the number of gaussian (kernel)
        self.gaussN = self.interN + 1 
        self.interL= 1. / self.interN
        self.posX = np.zeros([self.gaussN, DOF]) # gaussian avarage value
        self.rewX = np.zeros([self.gaussN, 2]) # gaussian avarage value
        self.std_dev = 0.031/3 #/ ((self.interN-1) * 2)
        
        # computate gaussian's average values
        for gn in xrange(self.gaussN):
            if gn == 0 :
                self.posX[0] = 0    
            else:
                self.posX[gn] = self.posX[gn-1] + self.interL
                      
        # computate gaussian's average values
        for gn in xrange(self.gaussN):
            if gn == 0 :
                self.rewX[0] = 0    
            else:
                self.rewX[gn] = self.rewX[gn-1] + self.interL* 10
        
        # TIME PARAMETERS
        self.delT = 0.1
        self.tau = 1.  
        self.trialN = 50000
        self.trialMov = 50
    
        # input units
        self.inputN = self.gaussN
        
        # output units
        self.actNOut = DOF
        self.critNOut = 1
        
        # learning parameters
        self.ACT_ETA =   1 * 10 **(-2)
        self.CRIT_ETA =  2 * 10 ** (-4)
        self.DISC_FACT = 5 * 10 ** (-6)
    
        # ANN's state arrays
        self.actState = np.zeros([self.gaussN * (DOF+2)])
        self.prvState = np.zeros([self.gaussN * (DOF+2)])
        self.stateBuff = np.zeros([self.gaussN * (DOF+2), self.trialMov])
        
        # weights
        self.actW = np.zeros([ self.actNOut, self.inputN * (DOF+2)])
        self.critW= np.zeros(self.inputN * (DOF+2))
        
        # noise
        self.actNoise = np.zeros(self.actNOut)
        self.prvNoise = np.zeros(self.actNOut)
        self.noiseBuff = np.zeros([DOF, self.trialMov])
        
        
        self.actOut = np.zeros(DOF)
        self.outBuff = np.zeros([DOF, self.trialMov])
        self.ep3D = np.zeros(DOF)
    
        # critic output parameters
        self.actRew = 0
        self.surp = 0.
        self.surpBuff = np.zeros(self.trialMov)
        self.actCritOut= np.zeros(self.critNOut)
        self.prvCritOut= np.zeros(self.critNOut)
        
        self.rewMov = np.zeros(self.trialN)
        self.trial200 = np.zeros(200)
        self.avgMov = np.zeros(self.trialN/200)
               
    def spreading(self, w, pattern):
        return np.dot(w, pattern)

    def computeTdError(self):
        x = self.DISC_FACT * self.actCritOut - self.prvCritOut
        return x + self.actRew 
        
    def trainAct(self):    
        return self.ACT_ETA * self.surp * np.outer(self.prvState, self.prvNoise)

    def trainActNONO(self):    
        return self.ACT_ETA * self.surp * self.prvState

        
    def trainCrit(self):
        return self.CRIT_ETA * self.surp * self.prvState