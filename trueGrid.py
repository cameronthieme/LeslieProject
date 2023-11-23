#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:47:36 2023

@author: cameronthieme

This code gives the Conley-Morse graph of every element in the true grid

Works after being called by shell file TrueRunner.sh
"""

# Section 1: parameters we play with

# choose number of subdivisions 
phase_subdiv = 18
print('Using %s subdivisions' %phase_subdiv)


# define boundaries of the problem
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]


# Section 2: Importing packages and determining parameter values

import CMGDB
import math
import time
import sys

from leslieFunctions import *

# timing everything 
startTime = time.time() # for computing runtime 
latestTime = startTime

# finding correct row/column in grid
job = int(sys.argv[1]) # Input file name
# job = 257

# parameter range from Database Paper,  and # boxes in each dim
th1max = 37
th1min = 8
N1 = 40
th2max = 50
th2min = 3
N2 = 40


# Can only batch 400 jobs
# Must get through 1600 parameter combos (grid is 40x40)
# each job does 5 parameter combos
for round in range(4):
    boxNum = job + round*400

    # box x,y coords
    xIter = boxNum % N1
    yIter = math.floor(boxNum/N1) % N2
    
    # theta values of box
    th1 = xIter * (th1max - th1min) / N1 + (th1max - th1min) / (2 * N1) + th1min
    th2 = yIter * (th2max - th2min) / N2 + (th2max - th2min) / (2 * N2) + th2min
        
    # names for saving files
    fileName = '_th1_' + str(th1) + '_th2_' + str(th2)
    fileNameMGtrue = 'MorseGraphs/MG_True' + fileName
    
    # Running CMGDB
    def LeslieIntervalBox(rect):
        return IntervalBoxMap2D(leslieInterval, rect, th1, th2)
    modelTrue = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, LeslieIntervalBox)
    morse_graph_True, map_graph_True = CMGDB.ComputeConleyMorseGraph(modelTrue)
    
    # saving Morse Graph
    mgTrue = CMGDB.PlotMorseGraph(morse_graph_True)
    mgTrue.save(fileNameMGtrue)


# Recording Execution Times
executionTime = time.time() - startTime
print('Total execution Time:')
print('Execution time, in seconds: ' + str(executionTime))
print('Execution time, in minutes: ' + str(executionTime/60))

