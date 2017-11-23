# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:03:40 2017

@author: Alex
"""
import numpy as np
import matplotlib as plt
import shapely.geometry as geo
import random as stdrandom
import math_utils as utils
    
class Maze8Arm:
        
    def init(self, returnGoal):
        
        #  MAZE BOUNDS
        self.maze8ArmPol = geo.Polygon([ (4.5,6.2) , (4.5,9.5) , (5.5,9.5) ,(5.5,6.2) , (8,8.7) , (8.7,8) , (6.2,5.5), (9.5,5.5),(9.5,4.5), (6.2,4.5),(8.7,2),(8,1.3),(5.5,3.8), (5.5,0.5), (4.5,0.5),(4.5,3.8), (2,1.3), (1.3,2),  (3.8,4.5),(0.5,4.5), (0.5,5.5), (3.8,5.5),(1.3,8), (2,8.7)])
        self.maze8arm_int =geo.LinearRing([ (4.53,6.17) , (4.53,9.47) , (5.47,9.47), (5.53,6.17) , (8,8.67) , (8.67,8),(6.17,5.47), (9.47,5.47),(9.47,4.53),(6.17,4.53),(8.67,2.),(8,1.33), (5.53,3.83), (5.47,0.53), (4.53,0.53),(4.53,3.83), (2.,1.33), (1.33,2.),(3.83,4.53),(0.53,4.53), (0.53,5.47),(3.83,5.47),(1.33,8), (2,8.67)])
        
        self.nArmPol = geo.Polygon([(4.5,6.2) , (4.5,9.5) , (5.5,9.5) , (5.5, 6.2)])    
        self.nArmInt = geo.LinearRing([(4.53,6.17) , (4.53,9.47) , (5.47,9.47) , (5.47, 6.17)])     
        
        self.neArmPol  = geo.Polygon([(5.5,6.2) , (8,8.7) , (8.7,8), (6.2,5.5)])     
        self.neArmInt  = geo.LinearRing([(5.53,6.17) , (8,8.67) , (8.67,8), (6.17,5.53)])     
 
        self.eArmPol  = geo.Polygon([(6.2,5.5), (9.5,5.5),(9.5,4.5), (6.2,4.5)])
        self.eArmInt= geo.LinearRing([(6.17,5.47), (9.47,5.47),(9.47,4.53), (6.17,4.53)])
   
        self.seArmPol  = geo.Polygon([(6.2,4.5),(8.7,2),(8,1.3), (5.5,3.8)])
        self.seArmInt= geo.LinearRing([(6.17,4.53),(8.67,2.),(8,1.33), (5.53,3.83)])
    
        self.sArmPol  = geo.Polygon([(5.5,3.8), (5.5,0.5), (4.5,0.5), (4.5,3.8)])
        self.sArmInt= geo.LinearRing([(5.47,3.83), (5.47,0.53), (4.53,0.53), (4.53,3.83)])
    
        self.swArmPol  = geo.Polygon([(4.5,3.8), (2,1.3), (1.3,2), (3.8,4.5)])
        self.swArmInt= geo.LinearRing([(4.53,3.83), (2.,1.33), (1.33,2), (3.83,4.53)])
    
        self.wArmPol  = geo.Polygon([(3.8,4.5),(0.5,4.5), (0.5,5.5), (3.8,5.5)])
        self.wArmInt= geo.LinearRing([(3.83,4.53),(0.53,4.53), (0.53,5.47), (3.83,5.47)])
    
        self.nwArmPol  = geo.Polygon([(3.8,5.5),(1.3,8), (2,8.7), (4.5,6.2)])
        self.nwArmInt= geo.LinearRing([(3.83,5.47),(1.33,8), (2,8.67), (4.47,6.17)])
    
        self.centerPol = geo.Polygon([(5.5, 6.2), (6.2,5.5),(6.2,4.5),(5.5,3.8),(4.5,3.8),(3.8,4.5),(3.8,5.5),(4.5,6.2)])
        self.centerInt= geo.LinearRing([(5.47, 6.17), (6.17,5.47),(6.17,4.53),(5.47,3.83),(4.53,3.83),(3.83,4.53),(3.83,5.47),(4.53,6.17)])
        
        self.rewList = [([5. , 9.24]),\
                        ([5. , 5.]),\
                        ([8. , 8.]),\
                        ([9.24 , 5.]),\
                        ([8. , 2.]),\
                        ([5. , 0.76]),\
                        ([2. , 2.]),\
                        ([0.76 , 5.]),\
                        ([2. , 8.]),]
        
        
        # REWARD & ANGENT POSITION
        if returnGoal == False:
            self.rewPosList = [([5. , 9.24]),\
                               ([8. , 8.]),\
                               ([9.24 , 5.]),\
                               ([8. , 2.]),\
                               ([5. , 0.76]),\
                               ([2. , 2.]),\
                               ([0.76 , 5.]),\
                               ([2. , 8.]),]
        else:
            self.rewPosList = [([5. , 9.24]),\
                               ([5. , 5.]),\
                               ([8. , 8.]),\
                               ([5. , 5.]),\
                               ([9.24 , 5.]),\
                               ([5. , 5.]),\
                               ([8. , 2.]),\
                               ([5. , 5.]),\
                               ([5. , 0.76]),\
                               ([5. , 5.]),\
                               ([2. , 2.]),\
                               ([5. , 5.]),\
                               ([0.76 , 5.]),\
                               ([5. , 5.]),\
                               ([2. , 8.]),\
                               ([5. , 5.]),]
        
        self.rewPos = np.zeros(2)
        self.distance = 0
        self.curr2DPos = np.ones(2) * 5.
        self.delta2DPos = np.zeros(2)
        self.next2DPos = np.zeros(2)
        
    #    self.posBuff = np.zeros([2,maxStep])
     #   self.trajectories = np.zeros([2, maxStep, maxTrial])
        
    
    
    
    
    def randomGoal(self):
        
        self.rewPos= stdrandom.choice([(5. , 9.24),\
                                       (8. , 8.),\
                                       (9.24 , 5.),\
                                       (8. , 2.),\
                                       (5. , 0.76),\
                                       (2. , 2.),\
                                       (0.76 , 5.),\
                                       (2. , 8.),]) 
            
    def setGoal(self, trial):
             
        if trial % 8 == 0:
            self.rewPos = ([5. , 9.24])
        elif trial % 8  == 1:
            self.rewPos = ([8. , 8.])
        elif trial % 8  == 2:
            self.rewPos = ([9.24 , 5.])
        elif trial % 8  == 3:
            self.rewPos = ([8. , 2.])
        elif trial % 8  == 4:
           self.rewPos = ([5. , 0.76])
        elif trial % 8  == 5:
            self.rewPos = ([2. , 2.])
        elif trial % 8  == 6:
            self.rewPos = ([0.76 , 5.])
        elif trial % 8  == 7:
            self.rewPos = ([2. , 8.])
            
            
    def setReturnGoal(self, trial):
        
        if trial % 16 == 0:
            self.rewPos = ([5. , 9.24])
        elif trial % 2 == 1:
            self.rewPos = ([5. , 5.])    
        elif trial % 16  == 2:
            self.rewPos = ([8. , 8.])
        elif trial % 16  == 4:
            self.rewPos = ([9.24 , 5.])
        elif trial % 16  == 6:
            self.rewPos = ([8. , 2.])
        elif trial % 16  == 8:
           self.rewPos = ([5. , 0.76])
        elif trial % 16  == 10:
            self.rewPos = ([2. , 2.])
        elif trial % 16  == 12:
            self.rewPos = ([0.76 , 5.])
        elif trial % 16  == 14:
            self.rewPos = ([2. , 8.])        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def computeDistance(self):
        self.distance = utils.distance2D(self.curr2DPos,self.rewPos)
        
    
    def imposeOutBounds(self): 
            
        self.agentPosition = geo.Point(self.curr2DPos)
        self.agentNextPosition = geo.Point(self.next2DPos)
            
        if self.maze8ArmPol.contains(self.agentNextPosition) == False:
                    
            if self.centerPol.contains(self.agentPosition) == True:
                self.d = self.centerInt.project(self.agentNextPosition)
                self.p = self.centerInt.interpolate(self.d)
                    
            elif self.nArmPol.contains(self.agentPosition) == True:
                self.d = self.nArmInt.project(self.agentNextPosition)
                self.p = self.nArmInt.interpolate(self.d)
                   
            elif self.neArmPol.contains(self.agentPosition) == True:
                self.d = self.neArmInt.project(self.agentNextPosition)
                self.p = self.neArmInt.interpolate(self.d)
                         
    
            elif self.eArmPol.contains(self.agentPosition) == True:
                self.d = self.eArmInt.project(self.agentNextPosition)
                self.p = self.eArmInt.interpolate(self.d) 
                    
 
            elif self.seArmPol.contains(self.agentPosition) == True:
                self.d = self.seArmInt.project(self.agentNextPosition)
                self.p = self.seArmInt.interpolate(self.d)
                    

            elif self.sArmPol.contains(self.agentPosition) == True:
                self.d = self.sArmInt.project(self.agentNextPosition)
                self.p = self.sArmInt.interpolate(self.d)
                    
            elif self.swArmPol.contains(self.agentPosition) == True:
                self.d = self.swArmInt.project(self.agentNextPosition)
                self.p = self.swArmInt.interpolate(self.d)  
                    
                
            elif self.wArmPol.contains(self.agentPosition) == True:
                self.d = self.wArmInt.project(self.agentNextPosition)
                self.p = self.wArmInt.interpolate(self.d)
                    
                
            elif self.nwArmPol.contains(self.agentPosition) == True:
                self.d = self.nwArmInt.project(self.agentNextPosition)
                self.p = self.nwArmInt.interpolate(self.d)    
                   
                                                    
            self.closestpoint = list(self.p.coords)[0]
            self.next2DPos = np.array(self.closestpoint).copy()
            
                
                        
    def imposeInBounds(self):
        
        self.agentPosition = geo.Point(self.curr2DPos)
        self.agentNextPosition = geo.Point(self.next2DPos)
        
        
        
        if self.nArmPol.contains(self.agentPosition) == True:
            if self.nArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.nArmInt.project(self.agentNextPosition)
                    self.p = self.nArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                    
        elif self.neArmPol.contains(self.agentPosition) == True:
            if self.neArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.neArmInt.project(self.agentNextPosition)
                    self.p= self.neArmInt.interpolate(self.dPol) 
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                    
        elif self.eArmPol.contains(self.agentPosition) == True:
            if self.eArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.eArmInt.project(self.agentNextPosition)
                    self.p = self.eArmInt.interpolate(self.dPol)        
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                    
        elif self.seArmPol.contains(self.agentPosition) == True:
            if self.seArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.seArmInt.project(self.agentNextPosition)
                    self.p = self.seArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                   
        elif self.sArmPol.contains(self.agentPosition) == True:
            if self.sArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.sArmInt.project(self.agentNextPosition)
                    self.p = self.sArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                   
        elif self.swArmPol.contains(self.agentPosition) == True:
            if self.swArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.swArmInt.project(self.agentNextPosition)
                    self.p = self.swArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]    
                    self.next2DPos = np.array(self.closestpoint).copy()
                   
        elif self.wArmPol.contains(self.agentPosition) == True:
            if self.wArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.wArmInt.project(self.agentNextPosition)
                    self.p = self.wArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                   
        elif self.nwArmPol.contains(self.agentPosition) == True:
            if self.nwArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.dPol = self.nwArmInt.project(self.agentNextPosition)
                    self.p = self.nwArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    self.next2DPos = np.array(self.closestpoint).copy()
                        
    def verifyOutside(self):
        
        self.agentNextPosition = geo.Point(self.next2DPos)
        
        if self.maze8ArmPol.contains(self.agentNextPosition) == False:
            self.outside = 1
        else:
            self.outside = 0
    
    
    
    
    
            
            
            
            
    def placeRewardPhases(self, phases):
        
        if phases == 0:
            self.rewPos = ([5 , 9.24])
        elif phases % 2  == 1:
            self.rewPos = ([5 , 5])    
        elif phases % 16  == 2:
            self.rewPos = ([8 , 8])    
        elif phases % 16  == 4:
            self.rewPos = ([9.24 , 5])
        elif phases % 16  == 6:
            self.rewPos = ([8 , 2])
        elif phases % 16  == 8:
            self.rewPos = ([5 , 0.76])
        elif phases % 16  == 10:
            self.rewPos = ([2 , 2])
        elif phases % 16  == 12:
            self.rewPos = ([0.76 , 5])
        elif phases % 16  == 14:
            self.rewPos = ([2 , 8])
        