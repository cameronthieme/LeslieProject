#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates Morse graphs for various parameter values
Domain sizes are dependent on parameters

@author: cameronthieme
"""

# Section 1: parameters we play with

# choose number of subdivisions 
phase_subdiv = 18
print('Using %s subdivisions' %phase_subdiv)

# choose training sizes for GP
sample_size = 500
print('Using %s training size' %sample_size)

# number of trials per parameter value
trials = 50

# choose length of trajectories used
# 1 means we sample uniformly in domain
traj_len = 1

# Choose to do measurement error ('meas_err') or step error ('step_err')
noise_type = 'meas_err'
print(noise_type)

# number of restarts for GP optimizer
n_restarts = 40
# choose RBF parameter for GP regression
tau = 6.445
beta = 1



# noise value
noise_std = 0.1

# how many jobs we can queue at once
batch_size = 400


# Section 2: Importing packages and determining parameter values

import CMGDB
import math
import time

import csv
import sys

from leslieFunctions import *

# timing everything 
startTime = time.time() # for computing runtime 
latestTime = startTime

# parameter range from Database Paper,  and # boxes in each dim
th1max = 37
th1min = 8
N1 = 40
th2max = 50
th2min = 3
N2 = 40

# total number of nodes we need to compute at
grid_size = N1*N2

# finding correct row/column in grid
job = int(sys.argv[1]) # Input file name
# job = 2

# determining total number of nodes for this job
if job < (grid_size % batch_size):    
    total_nodes = math.ceil(grid_size / batch_size)
else:
    total_nodes = math.floor(grid_size / batch_size)

for round in range(total_nodes):
    
    # determining parameter square
    node = job + round*batch_size    
    xIter = node % N1
    yIter = math.floor(node/N1) % N2
    
    # theta value
    th1 = xIter * (th1max - th1min) / N1 + (th1max - th1min) / (2 * N1) + th1min
    th2 = yIter * (th2max - th2min) / N2 + (th2max - th2min) / (2 * N2) + th2min
    print('th1: %s' %th1)
    print('th2: %s' %th2)

    # define boundaries of the problem
    lower_bounds = [-0.001, -0.001]
    upper_bounds = [10 * (th1 + th2) * math.exp(-1), 7 * (th1 + th2) * math.exp(-1)]

    # Section 3: Getting True Conley-Morse Info
    
    # names for saving files
    fileName = '_th1_' + str(th1) + '_th2_' + str(th2)
    
    for trial in range(trials):
        
        # train GP until we don't get a zero map
        zero_counter = 1
        fail_counter = 0
        while zero_counter == 1:     
            
            # Sampling training data
            x_train, y_train0, y_train1 = sampGenLes(
                noise_std = noise_std,
                sample_size = sample_size,
                noise_type = noise_type,
                traj_len = traj_len,
                th1 = th1,
                th2 = th2,
                lower_bounds = lower_bounds,
                upper_bounds = upper_bounds)
            
            # updating upper bounds
            upper_bounds_updated = [ max(upper_bounds[0], max(y_train0)*1.1), max(upper_bounds[1], max(y_train1)*1.1) ]
            
            # Train a GP with the data above
            gp0 = GP(X_train = x_train,
                     Y_train = y_train0,
                     tau = tau,
                     beta = beta,
                     noise_std = noise_std,
                     n_restarts = n_restarts)
            gp1 = GP(X_train = x_train,
                     Y_train = y_train1,
                     tau = tau,
                     beta = beta,
                     noise_std = noise_std,
                     n_restarts = n_restarts)
            
            # check if map becomes zero map
            zero_counter = zero_checker(gp0 = gp0,
                                        gp1 = gp1,
                                        lower_bounds = lower_bounds,
                                        upper_bounds = upper_bounds)
            
            # too many fails and we move on
            if zero_counter == 1:
                fail_counter += 1
            if (fail_counter > 5):
                break

            
            
        # Section 4: Running CMGDB & Checking Isomorphism
            
        # 4(a): running CMGDB
        # this is most time-consuming part by far (if we are using reasonable subdivs)
        #CMGDB computations
        
        # defining model parameters
        # True Morse Graph
        def LeslieIntervalBox(rect):
            return IntervalBoxMap2D(leslieInterval, rect, th1, th2)
        modelTrue = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, LeslieIntervalBox)
        # GP Morse Graph
        def GP_Box(rect):
            return BoxMapSD(rect, gp0, gp1)
        modelGP = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds_updated, GP_Box)
        
        # running the hard computations
        morse_graph_True, map_graph_True = CMGDB.ComputeConleyMorseGraph(modelTrue)
        morse_graph_GP, map_graph_GP = CMGDB.ComputeConleyMorseGraph(modelGP)
        
        # saving Morse Graphs
        fileNameMGtrue = 'MorseGraphs/MG_True' + fileName + '_trial' + str(trial)
        mgTrue = CMGDB.PlotMorseGraph(morse_graph_True)
        mgTrue.save(fileNameMGtrue)
        fileNameMGgp = 'MorseGraphs/MG_GP' + fileName + '_trial' + str(trial)
        mgGP = CMGDB.PlotMorseGraph(morse_graph_GP)
        mgGP.save(fileNameMGgp)

        
        # Section 5: Saving Data
        
        # Naming File
        fileNameResults = 'ResultTables/results'  + fileName + '_trial' + str(trial) + '.csv'
    
        # Info we want to save
        infoList = [th1, th2, zero_counter]
        
        # Writing info to CSV
        with open(fileNameResults, mode='w', newline='') as file:
            # Create a writer object
            writer = csv.writer(file)
            # Write the list to the CSV file
            writer.writerow(infoList)


# Recording Execution Times
executionTime = time.time() - startTime
print('Total execution Time:')
print('Execution time, in seconds: ' + str(executionTime))
print('Execution time, in minutes: ' + str(executionTime/60))










