# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 22:35:37 2017

@author: Alex
"""

import numpy as np
import math_utils as utils

class Cerebellum():
    
    def init(self, multiNet, stateBg, visionState, wristState, rewList, activeSistems, DOF = 3):
        
        # TIME
        self.DELTM = 0.1
        self.TAU = 1.
        
        # LEARNING
        self.cbETA = 0.04
        
        # STATE
        self.currState = np.zeros(len(stateBg))
        self.fwdVisionState = np.ones(len(np.hstack([visionState,wristState])))
        self.prvFwdVisionState = self.fwdVisionState.copy()
        
        # WEIGHTS
        self.w = np.zeros([len(stateBg), DOF])
        self.fwdVisionW = np.zeros([len(np.hstack([visionState,wristState])), 2])
        self.fwdWristW = np.zeros([len(np.hstack([wristState,wristState])), 3])
        
        if multiNet == True:            
            self.multiCerebW = np.zeros([len(stateBg), DOF, len(rewList)])
            self.multiFwdVisionW = np.zeros([len(np.hstack([visionState,wristState])), len(visionState), len(rewList)])
            self.multiFwdWristW = np.zeros([len(np.hstack([wristState,wristState])), len(wristState), len(rewList)])
        
         
        # TEACHING 
        self.trainOut = np.array([0.5,0.5,0.5])
        self.errorOut = np.zeros(DOF)
        
        self.trainEstVision = np.zeros(2)
        self.errorTrainEstVision = np.zeros(2)
        self.estVision = np.zeros(2)
        self.errorEstVision = np.zeros(2)
        self.fwdVisionError = np.zeros(2)
        self.prvFwdVisionError = np.zeros(2)
        
     #   self.trainEstWrist = np.ones(len(np.hstack([wristState,wristState]))) * 0.5
     #   self.errorEstWrist = np.zeros(len(wristState))
        
        # OUTPUT
        self.currOut = np.array([0.5,0.5,0.5])
        self.prvOut = np.array([0.5,0.5,0.5])
        
        
        self.estWrist = np.zeros(2)
                               
        self.trialFwdError = 0.
        
        
    def epochReset(self):
        
        # OUTPUT
        self.currOut = np.array([0.5,0.5,0.5])
        self.prvOut = np.array([0.5,0.5,0.5])
        
        self.estVision *= 0.
        self.errorEstVision *= 0
        
    def trialReset(self):
        
        # OUTPUT
        self.currOut = np.array([0.5,0.5,0.5])
        self.prvOut = np.array([0.5,0.5,0.5]) 
        
        self.trialFwdError *= 0 
        self.estVision *= 0.  
        self.errorEstVision *= 0            
        
    def trainCb(self,state, ep, T):
        self.trainOut = utils.sigmoid(np.dot(self.w.T, state))
        self.errorOut = ep - self.trainOut
        self.w += self.cbETA * np.outer(state, self.errorOut) * self.trainOut * (1. - self.trainOut) * T
                                       
    def trainCb2(self,state, ep, surp, T):
        self.trainOut = utils.sigmoid(np.dot(self.w.T, state))
        self.errorOut = ep - self.trainOut
        self.w += self.cbETA * np.outer(state, self.errorOut) * surp * self.trainOut * (1. - self.trainOut) * T                                   
                                       
        
    def trainFwdVision(self, curr2DPos):
        self.trainEstVision = utils.sigmoid(np.dot(self.fwdVisionW.T, self.prvFwdVisionState))
        self.errorTrainEstVision = utils.change_range(curr2DPos, -1., 1., 0., 1.)- self.trainEstVision
        self.fwdVisionW += self.cbETA * np.outer(self.prvFwdVisionState, self.errorTrainEstVision) 
    
        
    def spreading(self,state):
        self.currOut = utils.sigmoid(np.dot(self.w.T, state))
        
    def fwdVision(self):
        self.estVision = utils.sigmoid(np.dot(self.fwdVisionW.T, self.fwdVisionState))
   
        
        
        
        