# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:03:40 2017

@author: Alex
"""
import numpy as np
import matplotlib as plt
import shapely.geometry as geo

    
class Maze8Arm:
        
    def init(self):
        
        # POLYGONS
        self.maze8ArmPol = geo.Polygon([ (4.5,6.2) , (4.5,9.5) , (5.5,9.5) ,(5.5,6.2) , (8,8.7) , (8.7,8) , (6.2,5.5), (9.5,5.5),(9.5,4.5), (6.2,4.5),(8.7,2),(8,1.3),(5.5,3.8), (5.5,0.5), (4.5,0.5),(4.5,3.8), (2,1.3), (1.3,2),  (3.8,4.5),(0.5,4.5), (0.5,5.5), (3.8,5.5),(1.3,8), (2,8.7)])
        self.maze8arm_int =geo.LinearRing([ (4.53,6.17) , (4.53,9.47) , (5.47,9.47), (5.53,6.17) , (8,8.67) , (8.67,8),(6.17,5.47), (9.47,5.47),(9.47,4.53),(6.17,4.53),(8.67,2.),(8,1.33), (5.47,3.83), (5.47,0.53), (4.53,0.53),(4.53,3.83), (2.,1.33), (1.33,2.),(3.83,4.53),(0.53,4.53), (0.53,5.47),(3.83,5.47),(1.33,8), (2,8.67)])
        
        self.nArmPol = geo.Polygon([(4.5,6.2) , (4.5,9.5) , (5.5,9.5) , (5.5, 6.2)])    
        self.nArmInt = geo.LinearRing([(4.53,6.17) , (4.53,9.47) , (5.47,9.47) , (5.47, 6.17)])     
        
        self.neArmPol  = geo.Polygon([(5.5,6.2) , (8,8.7) , (8.7,8), (6.2,5.5)])     
        self.neArmInt  = geo.LinearRing([(5.53,6.17) , (8,8.67) , (8.67,8), (6.17,5.53)])     
 
        self.eArmPol  = geo.Polygon([(6.2,5.5), (9.5,5.5),(9.5,4.5), (6.2,4.5)])
        self.eArmInt= geo.LinearRing([(6.17,5.47), (9.47,5.47),(9.47,4.53), (6.17,4.53)])
   
        self.seArmPol  = geo.Polygon([(6.2,4.5),(8.7,2),(8,1.3), (5.5,3.8)])
        self.seArmInt= geo.LinearRing([(6.17,4.53),(8.67,2.),(8,1.33), (5.52,3.77)])
    
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
            
        self.rewPos = np.zeros(2)
        
    def placeReward(self, trial):
    
        
    
        if trial % 8 == 0:
            self.rewPos = np.array([5 , 9])
        if trial % 8 == 1:
            self.rewPos = np.array([8 , 8])
        if trial % 8 == 2:
            self.rewPos = np.array([9 , 5])
        if trial % 8 == 3:
            self.rewPos = np.array([8 , 2])
        if trial % 8 == 4:
            self.rewPos = np.array([5 , 1])
        if trial % 8 == 5:
            self.rewPos = np.array([2 , 2])
        if trial % 8 == 6:
            self.rewPos = np.array([1 , 5])
        if trial % 8 == 7:
            self.rewPos = np.array([2 , 8])
        
        
        
    
    

    
    def imposeLBounds(self, actualPosition, nextPosition): 
            
        self.agentPosition = geo.Point(actualPosition)
        self.agentNextPosition = geo.Point(nextPosition)
            
        if self.maze8ArmPol.contains(self.agentNextPosition) == False:
            
            self.outside = 1
                    
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
            nextPosition = np.array(self.closestpoint)
            
                
        return nextPosition
                        
    def imposeHbounds(self , actualPosition, nextPosition):
        
        self.agentPosition = geo.Point(actualPosition)
        self.agentNextPosition = geo.Point(nextPosition)
        
        
        
        if self.nArmPol.contains(self.agentPosition) == True:
            if self.nArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.nArmInt.project(self.agentNextPosition)
                    self.p = self.nArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]
                    nextPosition = np.array(self.closestpoint)
                    
        elif self.neArmPol.contains(self.agentPosition) == True:
            if self.neArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.neArmInt.project(self.agentNextPosition)
                    self.p= self.neArmInt.interpolate(self.dPol)                
                    self.closestpoint = list(self.p.coords)[0]
                    nextPosition = np.array(self.closestpoint)
                    
        elif self.eArmPol.contains(self.agentPosition) == True:
            if self.eArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.eArmInt.project(self.agentNextPosition)
                    self.p = self.eArmInt.interpolate(self.dPol)        
                    self.closestpoint = list(self.p.coords)[0]
                    nextPosition = np.array(self.closestpoint)
                    
        elif self.seArmPol.contains(self.agentPosition) == True:
            if self.seArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.seArmInt.project(self.agentNextPosition)
                    self.p = self.seArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]    
                    nextPosition = np.array(self.closestpoint)
                   
        elif self.sArmPol.contains(self.agentPosition) == True:
            if self.sArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.sArmInt.project(self.agentNextPosition)
                    self.p = self.sArmInt.interpolate(self.dPol)  
                    self.closestpoint = list(self.p.coords)[0]
                    nextPosition = np.array(self.closestpoint)
                   
        elif self.swArmPol.contains(self.agentPosition) == True:
            if self.swArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.swArmInt.project(self.agentNextPosition)
                    self.p = self.swArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0]    
                    nextPosition = np.array(self.closestpoint)
                   
        elif self.wArmPol.contains(self.agentPosition) == True:
            if self.wArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.wArmInt.project(self.agentNextPosition)
                    self.p = self.wArmInt.interpolate(self.dPol)    
                    self.closestpoint = list(self.p.coords)[0] 
                    nextPosition = np.array(self.closestpoint)
                   
        elif self.nwArmPol.contains(self.agentPosition) == True:
            if self.nwArmPol.contains(self.agentNextPosition) == False:
                if self.centerPol.contains(self.agentNextPosition) == False:
                    self.outside =1
                    self.dPol = self.nwArmInt.project(self.agentNextPosition)
                    self.p = self.nwArmInt.interpolate(self.dPol)
                    self.closestpoint = list(self.p.coords)[0] 
                    nextPosition = np.array(self.closestpoint)
                    
        
        return nextPosition
                        
    def verifyOutside(self,nextPosition):
        
        self.agentNextPosition = geo.Point(nextPosition)
        
        if self.maze8ArmPol.contains(self.agentNextPosition) == False:
            self.outside = 1
        else:
            self.outside = 0
    
    
    
    def plot2Dinit(rewardPosition, actual2DPosition, actual3DPosition):
        
        #♣ INIT plotting         
        fig1   = plt.figure("Workspace",figsize=(80,80), )
        gs = plt.GridSpec(1, 2, width_ratios=[3, 3]) 
        # add counters
        text1 = plt.figtext(.9, .1, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
        text2 = plt.figtext(.1, .9, "movement = %s" % (0), style='italic', bbox={'facecolor':'red'})
             
        # MAZE COORDINATES
                       
        # north track
        line1 = plt.Line2D([4.5,4.5] , [6.2,9.5] , color = 'black')
        line2 = plt.Line2D([4.5,5.5] , [9.5,9.5] , color = 'black')
        line3 = plt.Line2D([5.5,5.5] , [6.2,9.5] , color = 'black')
        line4 = plt.Line2D([4.5,5.5] , [6.2,6.2] , color = 'silver')
        circle1 = plt.Circle((5 , 9), 0.5, color = 'springgreen') 
        edgecircle1 = plt.Circle((5 , 9), 0.5, color = 'black', fill = False)
        # north east track 
        line5 = plt.Line2D([5.5,8], [6.2,8.7] , color = 'black')
        line6 = plt.Line2D([8,8.7], [8.7,8] , color = 'black')
        line7 = plt.Line2D([8.7,6.2], [8,5.5] , color = 'black')
        line8 = plt.Line2D([5.5,6.2], [6.2,5.5] , color = 'silver')   
        circle2 = plt.Circle((8 , 8), 0.5, color = 'springgreen')
        edgecircle2 = plt.Circle((8 , 8), 0.5, color = 'black', fill = False)  
        # east track   
        line9 = plt.Line2D([6.20,9.5], [5.5,5.5] , color = 'black')
        line10 = plt.Line2D([9.5,9.5], [5.5,4.5] , color = 'black')
        line11 = plt.Line2D([9.5,6.2], [4.5,4.5] , color = 'black')
        line12 = plt.Line2D([6.2,6.2], [5.5,4.5] , color = 'silver')
        circle3 = plt.Circle((9 , 5), 0.5, color = 'springgreen')
        edgecircle3 = plt.Circle((9 , 5), 0.5, color = 'black', fill = False)
        # south east track 
        line13 = plt.Line2D([6.2 , 8.7] , [4.5,2] , color = 'black')
        line14 = plt.Line2D([8.7,8] , [ 2, 1.3] , color = 'black')
        line15 = plt.Line2D([8 , 5.5], [1.3 , 3.8] , color = 'black')
        line16 = plt.Line2D([5.5,6.2], [3.8,4.5] , color = 'silver')
        circle4 = plt.Circle((8 , 2), 0.5, color = 'springgreen')
        edgecircle4 = plt.Circle((8 , 2), 0.5, color = 'black', fill = False)
        # south track  
        line17 = plt.Line2D([5.5,5.5], [3.8,0.5] , color = 'black')
        line18 = plt.Line2D([5.5,4.5], [0.5,0.5] , color = 'black')
        line19 = plt.Line2D([4.5,4.5], [0.5,3.8] , color = 'black')
        line20 = plt.Line2D([4.5,5.5], [3.8,3.8] , color = 'silver')
        circle5 = plt.Circle((5 , 1), 0.5, color = 'springgreen')
        edgecircle5 = plt.Circle((5 , 1), 0.5, color = 'black', fill = False)
        # south west track
        line21 = plt.Line2D([4.5,2], [3.8,1.3] , color = 'black')
        line22 = plt.Line2D([2,1.3], [1.3,2] , color = 'black')
        line23 = plt.Line2D([1.3,3.8], [2.,4.5] , color = 'black')
        line24 = plt.Line2D([4.5,3.8], [3.8,4.5] , color = 'silver')
        circle6 = plt.Circle((2 , 2), 0.5, color = 'springgreen')
        edgecircle6 = plt.Circle((2 , 2), 0.5, color = 'black', fill = False)
        # west track 
        line25 = plt.Line2D([3.8,0.5], [4.5,4.5] , color = 'black')
        line26 = plt.Line2D([0.5,0.5], [4.5,5.5] , color = 'black')
        line27 = plt.Line2D([0.5,3.8], [5.5,5.5] , color = 'black')
        line28 = plt.Line2D([3.8,3.8], [4.5,5.5] , color = 'silver')
        circle7 = plt.Circle((1 , 5), 0.5, color = 'springgreen')
        edgecircle7 = plt.Circle((1 , 5), 0.5, color = 'black', fill = False)
        # NORTH WEST TRACK
        line29 = plt.Line2D([3.8,1.3], [5.5,8] , color = 'black')
        line30 = plt.Line2D([1.3, 2], [8,8.7] , color = 'black')
        line31 = plt.Line2D([2.,4.5], [8.7,6.2] , color = 'black')
        line32 = plt.Line2D([3.8,4.5], [5.5,6.2] , color = 'silver')
        circle8 = plt.Circle((2 , 8), 0.5, color = 'springgreen')
        edgecircle8 = plt.Circle((2 , 8), 0.5, color = 'black', fill = False)
        
        # INIT 2d PLOTTING
        ax1 = fig1.add_subplot(gs[0])
        ax1.set_xlim([0,10])
        ax1.set_ylim([0,10]) 
        # north track
        ax1.add_artist(line1)
        ax1.add_artist(line2)
        ax1.add_artist(line3)
        ax1.add_artist(line4)
        ax1.add_artist(circle1)
        ax1.add_artist(edgecircle1)
        #north east track
        ax1.add_artist(line5)
        ax1.add_artist(line6)
        ax1.add_artist(line7)
        ax1.add_artist(line8)
        ax1.add_artist(circle2)
        ax1.add_artist(edgecircle2)
        #east track
        ax1.add_artist(line9)
        ax1.add_artist(line10)
        ax1.add_artist(line11)
        ax1.add_artist(line12)
        ax1.add_artist(circle3)
        ax1.add_artist(edgecircle3)
        #south east track
        ax1.add_artist(line13)
        ax1.add_artist(line14)
        ax1.add_artist(line15)
        ax1.add_artist(line16)
        ax1.add_artist(circle4)
        ax1.add_artist(edgecircle4)
        #south track
        ax1.add_artist(line17)
        ax1.add_artist(line18)
        ax1.add_artist(line19)
        ax1.add_artist(line20)
        ax1.add_artist(circle5)
        ax1.add_artist(edgecircle5)
        #south west track
        ax1.add_artist(line21)
        ax1.add_artist(line22)
        ax1.add_artist(line23)
        ax1.add_artist(line24)
        ax1.add_artist(circle6)
        ax1.add_artist(edgecircle6)
        #west track 
        ax1.add_artist(line25)
        ax1.add_artist(line26)
        ax1.add_artist(line27)
        ax1.add_artist(line28)
        ax1.add_artist(circle7)
        ax1.add_artist(edgecircle7)
        #north west track
        ax1.add_artist(line29)
        ax1.add_artist(line30)
        ax1.add_artist(line31)
        ax1.add_artist(line32)
        ax1.add_artist(circle8)
        ax1.add_artist(edgecircle8)
        #plot agent & reward
        reward, = ax1.plot(rewardPosition[0] , rewardPosition[1], 'x', color='r')
        agent, = ax1.plot(actual2DPosition[0], actual2DPosition[1], 'o')
        
        # INIT 3D PLOT
        ax2  = fig1.add_subplot(gs[1], projection='3d')
        # set limits
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])
        ax2.set_zlim([0,1])
        # set ticks
        ax2.set_xticks(np.arange(0, 1, 0.1))
        ax2.set_yticks(np.arange(0, 1, 0.1))
        ax2.set_zticks(np.arange(0, 1, 0.1))
        
        # plot agent
        agent3d, = ax2.plot([actual3DPosition[0]], [actual3DPosition[1]], [actual3DPosition[2]], 'o', color="blue")