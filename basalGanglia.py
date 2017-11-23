# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:36:46 2017

@author: Alex
"""

import numpy as np
import math_utils as utils

class ActorCritic:
    
    def init(self,  VISION, GOAL_VISION , AGENT_VISION, wrist,  multiNet, maxStep, wristRange, DOF = 3):
        
        
        #TIME PARAMETERS
        self.DELTM = 0.1
        self.TAU = 1.
        
        # LEARNING PARAMETeRS
        self.ACT_ETA = 0.5#0.1
        self.CRIT_ETA =  0.01 #0.01
        self.DISC_FACT = 0.9
        
        self.activeSistems = np.array([])
        
        # wrist paramenters
        if wrist == True:
            
            self.inputUnitsWrist= 125
            self.intervalsWrist = int(np.cbrt(self.inputUnitsWrist+1))
            self.sigWrist= 1. / ((self.intervalsWrist) * 2)
            self.wristGrid = utils.build3DGrid(self.intervalsWrist, 1)
            self.wristRawState = np.zeros([self.intervalsWrist,self.intervalsWrist,self.intervalsWrist])
            self.wristState= np.zeros(self.inputUnitsWrist)
            self.activeSistems =np.hstack([self.activeSistems,np.array([len(self.wristState)])])
     #       self.activeSistems.append(np.zeros(self.wristState))
        
        
        # VISION PARAMETERS
        if VISION == True:
            
            self.inputUnitsVision = 100
            self.intervalsVision = int(np.sqrt(self.inputUnitsVision+1))                                                                        #◘ 199 # define intervals used to define the number of gaussian (kernel)
            self.sigVision= 1. / ((self.intervalsVision) * 2) * 10
            self.visionGrid = utils.build2DGrid(self.intervalsVision, 10)
            self.visionRawState = np.zeros([self.intervalsVision,self.intervalsVision])
            self.visionState = np.array([])
            
            if GOAL_VISION == True:
                self.goalVisionState = np.zeros(self.inputUnitsVision)
            else:
                self.goalVisionState  = np.array([])
            self.visionState = np.hstack([self.visionState,self.goalVisionState])
            
            
            if AGENT_VISION == True:
                self.agentVisionState = np.zeros(self.inputUnitsVision)    
            else:
                self.agentVisionState  = np.array([])
            
            self.visionState = np.zeros(len(self.visionState)+len(self.agentVisionState))
        
            self.activeSistems= np.hstack([self.activeSistems,np.array([len(self.visionState)])])

        
        # GLOBAL PARAMETERS
        self.currState = np.zeros(int(np.sum(self.activeSistems)))
        self.prvState = np.zeros(int(np.sum(self.activeSistems)))
        
        
        self.stateBuff = np.zeros([int(np.sum(self.activeSistems)) , maxStep])
        self.desOutBuff = np.zeros([DOF, maxStep])
        self.surpBuff = np.zeros(maxStep)
     
        self.actW = np.zeros([int(np.sum((self.activeSistems))), DOF])
        self.critW= np.zeros(int(np.sum((self.activeSistems))))
        
            
        if multiNet == True:
            self.actW0 = self.actW.copy()
            self.actW1 = self.actW.copy()
            self.actW2 = self.actW.copy()
            self.actW3 = self.actW.copy()
            self.actW4 = self.actW.copy()
            self.actW5 = self.actW.copy()
            self.actW6 = self.actW.copy()
            self.actW7 = self.actW.copy()
            self.actW8 = self.actW.copy()
            
            self.critW0= self.critW.copy()
            self.critW1= self.critW.copy()
            self.critW2= self.critW.copy()
            self.critW3= self.critW.copy()
            self.critW4= self.critW.copy()
            self.critW5= self.critW.copy()
            self.critW6= self.critW.copy()
            self.critW7= self.critW.copy()
            self.critW8= self.critW.copy()
        
        # output
        self.currActOut = np.zeros(DOF)
        self.prvActOut = np.zeros(DOF)
        
        self.actRew = 0
        self.surp = 0
        self.currCritOut = np.zeros(1)
        self.prvCritOut =  np.zeros(1)
        
        self.currNoise = np.zeros(DOF)
        self.prvNoise = np.zeros(DOF)
        
    def computate_noise(self,T): 
        self.C1 = self.DELTM/ self.TAU
        self.C2 = 0.3
        self.currNoise = self.prvNoise + self.C1 * (self.C2 * np.random.randn(*self.prvNoise.shape) - self.prvNoise) * T
            
    def compGoalVisionState(self, goalPosition):
        self.goalVisionRawState = (utils.gaussian2D(goalPosition,self.visionGrid,self.sigVision,self.visionRawState,self.intervalsVision)).copy()
        self.goalVisionState = self.goalVisionRawState.reshape(self.inputUnitsVision).copy()
        
    def compAgentVisionState(self, agentPosition):
        self.agentVisionRawState = (utils.gaussian2D(agentPosition,self.visionGrid,self.sigVision,self.visionRawState,self.intervalsVision)).copy()
        self.agentVisionState = self.agentVisionRawState.reshape(self.inputUnitsVision).copy()
        
    def compWristState(self, wristPosition):
        self.wristRawState = (utils.gaussian3D(wristPosition,self.wristGrid,self.sigWrist,self.wristRawState,self.intervalsWrist)).copy()
        self.wristState = self.wristRawState.reshape(self.inputUnitsWrist).copy()      
    
    
    def spreadAct(self):
        self.currActOut[0] = utils.sigmoid(np.dot(self.actW[:,0], self.currState)).copy()
        self.currActOut[1] = utils.sigmoid(np.dot(self.actW[:,1], self.currState)).copy()
        self.currActOut[2] = utils.sigmoid(np.dot(self.actW[:,2], self.currState)).copy()
        
    
    def spreadCrit(self): 
        self.currCritOut = np.dot(self.critW, self.currState)
        

    def compSurprise(self):
        self.surp = (self.actRew + (self.DISC_FACT * self.currCritOut) - self.prvCritOut)
           
    def trainAct(self):    
        self.actW += np.outer(self.prvState, self.prvNoise) * self.surp * self.ACT_ETA

       
    def trainCrit(self):
        self.critW += self.CRIT_ETA * self.surp * self.prvState
        
        
    def trainAct2(self,prvCerebEp):
        self.actW += np.outer(self.prvState, (prvCerebEp - self.prvActOut)) * self.surp * self.ACT_ETA