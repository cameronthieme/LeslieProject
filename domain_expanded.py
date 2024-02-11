#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10, 2024

Takes a single data sample and 

@author: cameronthieme
"""

# Section 1: parameters we play with

# choose number of subdivisions 
phase_subdiv = 18

# number of restarts for GP optimizer
n_restarts = 40
# choose RBF parameter for GP regression
tau = 6.445
beta = 1

# no noise
noise_std = 0

# define initial boundaries of the problem
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]

# expanded boundaries
lower_bounds_expanded = [-0.001, -0.001]
upper_bounds_expanded = [150.0, 130.0]

# Section 2: Importing packages and determining parameter values

import CMGDB
import math
import time
import csv
import sys
from interval import interval, imath


from leslieFunctions import *

# parameters
th1 = 35
th2 = 22


# timing everything 
startTime = time.time() # for computing runtime 
latestTime = startTime

# Section 3: Regression

# Importing sample data
# Read the CSV file into a numpy array
xy_data = np.genfromtxt('sample_data.csv', delimiter=',', skip_header=1)

x_data = xy_data[:,:2]
y0_data = xy_data[:,2]
y1_data = xy_data[:,3]


# Train a GP with the data above
gp0 = GP(X_train = x_data,
            Y_train = y0_data,
            tau = tau,
            beta = beta,
            noise_std = noise_std,
            n_restarts = n_restarts)
gp1 = GP(X_train = x_data,
            Y_train = y1_data,
            tau = tau,
            beta = beta,
            noise_std = noise_std,
            n_restarts = n_restarts)

# check if map becomes zero map
zero_counter = zero_checker(gp0 = gp0,
                            gp1 = gp1,
                            lower_bounds = lower_bounds,
                            upper_bounds = upper_bounds)


# Section 4: Running CMGDB & Checking Isomorphism

# names for saving files
fileName = '_sample_data_'
fileNameMGtrue_initial_domain = 'MorseGraphs/MG_True' + fileName + 'initial_domain'
fileNameMGgp_initial_domain = 'MorseGraphs/MG_GP' + fileName + 'initial_domain'
fileNameMGtrue_expanded_domain = 'MorseGraphs/MG_True' + fileName + 'expanded_domain'
fileNameMGgp_expanded_domain = 'MorseGraphs/MG_GP' + fileName + 'expanded_domain'


if zero_counter == 0: # only triggers if mean nonzero
    
    # 4(a): running CMGDB
    # this is most time-consuming part by far (if we are using reasonable subdivs)
    #CMGDB computations
    
    # defining model parameters
    def LeslieIntervalBox(rect):
        return IntervalBoxMap2D(leslieInterval, rect, th1, th2)

    modelTrue_initial_domain = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, LeslieIntervalBox)
    modelTrue_expanded_domain = CMGDB.Model(phase_subdiv, lower_bounds_expanded, upper_bounds_expanded, LeslieIntervalBox)
    def GP_Box(rect):
        return BoxMapSD(rect, gp0, gp1)
    modelGP_initial_domain = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, GP_Box)
    modelGP_expanded_domain = CMGDB.Model(phase_subdiv, lower_bounds_expanded, upper_bounds_expanded, GP_Box)
    
    # running the hard computation
    morse_graph_True_expanded_domain, map_graph_True_expanded_domain = CMGDB.ComputeConleyMorseGraph(modelTrue_expanded_domain)
    morse_graph_GP_expanded_domain, map_graph_GP_expanded_domain = CMGDB.ComputeConleyMorseGraph(modelGP_expanded_domain)
    morse_graph_True_initial_domain, map_graph_True_initial_domain = CMGDB.ComputeConleyMorseGraph(modelTrue_initial_domain)
    morse_graph_GP_initial_domain, map_graph_GP_initial_domain = CMGDB.ComputeConleyMorseGraph(modelGP_initial_domain)
    
    # saving Morse Graphs
    mgTrue_initial_domain = CMGDB.PlotMorseGraph(morse_graph_True_initial_domain)
    mgTrue_initial_domain.save(fileNameMGtrue_initial_domain)
    mgGP_initial_domain = CMGDB.PlotMorseGraph(morse_graph_GP_initial_domain)
    mgGP_initial_domain.save(fileNameMGgp_initial_domain)
    mgTrue_expanded_domain = CMGDB.PlotMorseGraph(morse_graph_True_expanded_domain)
    mgTrue_expanded_domain.save(fileNameMGtrue_expanded_domain)
    mgGP_expanded_domain = CMGDB.PlotMorseGraph(morse_graph_GP_expanded_domain)
    mgGP_expanded_domain.save(fileNameMGgp_expanded_domain)
    

else:
    print('Failure to Train GP')

# Recording Execution Times
executionTime = time.time() - startTime
print('Total execution Time:')
print('Execution time, in seconds: ' + str(executionTime))
print('Execution time, in minutes: ' + str(executionTime/60))











