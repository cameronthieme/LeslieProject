#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:47:36 2023

@author: cameronthieme
"""

# Section 1: parameters we play with

# choose number of subdivisions 
phase_subdiv = 18
print('Using %s subdivisions' %phase_subdiv)

# choose length of trajectories used
# 1 means we sample uniformly in domain
# will later override this if we are using fullLenTraj
traj_len = 10
fullLenTraj = False
if fullLenTraj:
    print('Single Trajectory')
else:
    print('Trajectory Length ' + str(traj_len))

# embedding or no
embed = False
delay_num = 1 
print('Embedding: ' + str(embed))

# Choose to do measurement error ('meas_err') or step error ('step_err')
noise_type = 'step_err'
print(noise_type)


# number of restarts for GP optimizer
n_restarts = 40
# choose RBF parameter for GP regression
tau = 6.445
beta = 1

# define boundaries of the problem
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]
upper_bounds_trans = [90.0, 90.0]

# choose training sizes for GP (number rand initial conditions)
trainSizeList = [100, 200, 500, 1000, 2000]

# noise value
noiseLevels = [0,0.05, 0.1, 0.25, 0.5, 1, 1.5, 2]

# number of trials per combo
trials = 50

# Section 2: Importing packages and determining parameter values

import CMGDB
import math
import time
import csv
import sys

from leslieFunctions import *

# parapemeters
th1 = 19.6
th2 = 23.68

# timing everything 
startTime = time.time() # for computing runtime 
latestTime = startTime

# finding correct row/column in grid
job = int(sys.argv[1]) # Input file name
# job = 2
trial = job % trials
noise_num = math.floor(job/trials) % len(noiseLevels)
noise_std = noiseLevels[noise_num]




for train_size in trainSizeList:

    # if we want whole sample to be one (probably noisy) trajectory:
    if fullLenTraj:
        traj_len = train_size
    
    
    # Section 3: Regression
    
    # Sampling training data
    x_train, y_train0, y_train1 = sampGenLes(
        noise_std = noise_std,
        sample_size = train_size,
        noise_type = noise_type,
        traj_len = traj_len,
        th1 = th1,
        th2 = th2,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        delay_num = delay_num,
        embed = embed)
    
    
    
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
                                upper_bounds = upper_bounds_trans)
    
    
    # Section 4: Running CMGDB & Checking Isomorphism
    
    # names for saving files
    fileName = '_sampleSize' + str(train_size) + 'trajLen' + str(traj_len) + 'noise' + str(noise_std) + '_' + noise_type + '_embed_' + str(embed) + '_trial_' + str(trial)
    fileNameResults = 'ResultTables/results' + fileName  + '.csv'
    fileNameMGtrue = 'MorseGraphs/MG_True' + fileName
    fileNameMGgp = 'MorseGraphs/MG_GP' + fileName
    
    
    
    if zero_counter == 0: # only triggers if mean nonzero
        
        # 4(a): running CMGDB
        # this is most time-consuming part by far (if we are using reasonable subdivs)
        #CMGDB computations
        
        # defining model parameters
        def LeslieIntervalBox(rect):
            return IntervalBoxMap2D(leslieInterval, rect, th1, th2)
    
        modelTrue = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, LeslieIntervalBox)
        def GP_Box(rect):
            return BoxMapSD(rect, gp0, gp1)
        modelGP = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds_trans, GP_Box)
        
        # running the hard computation
        morse_graph_True, map_graph_True = CMGDB.ComputeConleyMorseGraph(modelTrue)
        morse_graph_GP, map_graph_GP = CMGDB.ComputeConleyMorseGraph(modelGP)
        
        # saving Morse Graphs
        mgTrue = CMGDB.PlotMorseGraph(morse_graph_True)
        mgTrue.save(fileNameMGtrue)
        mgGP = CMGDB.PlotMorseGraph(morse_graph_GP)
        mgGP.save(fileNameMGgp)
        
        # Section 4b: Checking For Nontrivial Graph Isomorphisms
        
        identical = ConleyMorseMatcher(morse_graph_True, morse_graph_GP)
    
    else:
        identical = 0
    
    # Section 5: Saving Data
    
    # Info we want to save
    infoList = [train_size, traj_len, noise_std, noise_type, embed, identical, zero_counter]
    
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











