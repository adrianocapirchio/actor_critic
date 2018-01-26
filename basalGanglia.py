# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 22:32:41 2017

@author: Alex
"""

import numpy as np
import math_utils as utils

class ActorCritic:
    
    def init(self,  VISION, GOAL_VISION , AGENT_VISION, WRIST,  multiNet, maxStep, wristRange, rewList, DOF = 3):
        
        
        # TIME PARAMETERS
        self.DELTM = 0.1
        self.TAU = 1.
        
        # LEARNING PARAMETERS
        self.ACT_ETA = 0.01#0.1
        self.CRIT_ETA =  0.001
        self.DISC_FACT = 0.9
        
        self.activeSistems = np.array([])
        
        # wrist paramenters
        if WRIST == True:
            
            self.inputUnitsWrist= 27
            self.intervalsWrist = int(np.cbrt(self.inputUnitsWrist+1))
            self.sigWrist= 1. / ((self.intervalsWrist) * 2)
            self.wristGrid = utils.build3DGrid(self.intervalsWrist, 1)
            self.wristRawState = np.zeros([self.intervalsWrist,self.intervalsWrist,self.intervalsWrist])
            self.wristState= np.zeros(self.inputUnitsWrist)
            self.activeSistems =np.hstack([self.activeSistems,np.array([len(self.wristState)])])
        
        
        # VISION PARAMETERS
        if VISION == True:
            
            self.inputUnitsVision = 25
            self.intervalsVision = int(np.sqrt(self.inputUnitsVision+1))                                                                        #◘ 199 # define intervals used to define the number of gaussian (kernel)
            self.sigVision= 1. / ((self.intervalsVision) * 2) 
            self.visionGrid = utils.build2DGrid(self.intervalsVision, 1)
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

        # WEIGHTS
        self.actW = np.zeros([int(np.sum((self.activeSistems))), DOF])
        self.critW= np.zeros(int(np.sum((self.activeSistems))))
        
        if multiNet == True:
            self.multiActW = np.zeros([int(np.sum((self.activeSistems))), DOF, len(rewList)])         
            self.multiCritW = np.zeros([int(np.sum((self.activeSistems))), len(rewList)])



        # STATE PARAMETERS
        self.currState = np.zeros(int(np.sum(self.activeSistems)))
        self.prvState = np.zeros(int(np.sum(self.activeSistems)))
        self.prv5State = np.zeros(int(np.sum(self.activeSistems)))
        
        
        
        self.stateBuff = np.zeros([int(np.sum(self.activeSistems)) , maxStep])
        self.desOutBuff = np.zeros([DOF, maxStep])
        self.surpBuff = np.zeros(maxStep)
        
        
        #◘ CRITIC PARAMETERS
        self.actRew = 0
        self.surp = 0.
        self.currCritOut = np.zeros(1)
        self.prvCritOut = np.zeros(1)
        
        # ACTOR PARAMETERS
        self.currActOut = np.array([0.5,0.5,0.5])
        self.trainOut = np.array([0.5,0.5,0.5])
        self.prvActOut = np.array([0.5,0.5,0.5])
        self.currNoise = np.zeros(DOF)
        self.prvNoise = np.zeros(DOF)
        
    def epochReset(self):
        
        # STATE PARAMETERS
        self.currState *= 0
        self.prvState *= 0
        self.prv5State *= 0
        
        #◘ CRITIC PARAMETERS
        self.actRew = 0
        self.surp *= 0 
        self.currCritOut *= 0
        self.prvCritOut *= 0

        # ACTOR PARAMETERS
        self.currActOut = np.array([0.5,0.5,0.5]) 
        self.trainOut = np.array([0.5,0.5,0.5])
        self.prvTrainOut = np.array([0.5,0.5,0.5])
        self.prvActOut = np.array([0.5,0.5,0.5])
        self.currNoise *= 0
        self.prvNoise *= 0
        
    def trialReset(self):
        
        # STATE PARAMETERS
        self.currState *= 0
        self.prvState *=0
        self.prv5State *= 0
        
        self.stateBuff *= 0
        self.desOutBuff *= 0
        self.surpBuff *= 0
        
        # CRITIC PARAMETERS
        self.actRew = 0
        self.currCritOut *= 0
        self.prvCritOut *= 0
        self.surp *= 0 
        
        # ACTOR PARAMETERS
        self.currActOut = np.array([0.5,0.5,0.5])
        self.trainOut = np.array([0.5,0.5,0.5])
        self.prvTrainOut = np.array([0.5,0.5,0.5])
        self.prvActOut = np.array([0.5,0.5,0.5])
        self.currNoise *= 0
        self.prvNoise *= 0 
        
    def compGoalVisionState(self, goalPosition):
        self.goalVisionRawState = (utils.gaussian2D(goalPosition,self.visionGrid,self.sigVision,self.visionRawState,self.intervalsVision)).copy()
        self.goalVisionState = self.goalVisionRawState.reshape(self.inputUnitsVision).copy()
        
    def compAgentVisionState(self, agentPosition):
        self.agentVisionRawState = (utils.gaussian2D(agentPosition,self.visionGrid,self.sigVision,self.visionRawState,self.intervalsVision)).copy()
        self.agentVisionState = self.agentVisionRawState.reshape(self.inputUnitsVision).copy()
        
    def compWristState(self, wristPosition):
        self.wristRawState = (utils.gaussian3D(wristPosition,self.wristGrid,self.sigWrist,self.wristRawState,self.intervalsWrist)).copy()
        self.wristState = self.wristRawState.reshape(self.inputUnitsWrist).copy()
        
    def compEpState(self, ep): 
        self.epRawState = (utils.gaussian3D(ep,self.wristGrid,self.sigWrist,self.wristRawState,self.intervalsWrist)).copy()
        self.epState = self.wristRawState.reshape(self.inputUnitsWrist).copy()
        
    def spreadAct(self):
        self.currActOut = utils.sigmoid(np.dot(self.actW.T, self.currState))
        
    def computate_noise(self,T): 
        self.C1 = self.DELTM/ self.TAU
        self.C2 = 1. - self.C1
        self.currNoise = (self.C2 * self.prvNoise + self.C1 *  np.random.normal(0., 0.6, 3)) * T
     #   self.currNoise = self.C1 * self.C2 * np.random.randn(3)
     #   self.currNoise = (self.C2 * self.prvNoise + (self.C1 * np.random.uniform(-0.5, 0.5, 3) - self.prvNoise)) * T
        
    def spreadCrit(self): 
        self.currCritOut = np.dot(self.critW, self.currState)
        
    def compSurprise(self):
        self.surp = self.actRew + (self.DISC_FACT * self.currCritOut) - self.prvCritOut
    







    
    def trainCrit(self):
        self.critW += self.CRIT_ETA * self.surp * self.prv5State
        
    def trainAct(self):    
        self.actW += np.outer(self.prv5State, self.prvNoise) * self.surp * self.ACT_ETA 
                               
    def trainAct2(self, cerebPrvOut, T):
        self.actW += np.outer(self.prvState, (cerebPrvOut- self.prvTrainOut)) * self.ACT_ETA * T
        
    
        
        
