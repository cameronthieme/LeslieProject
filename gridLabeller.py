#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:47:36 2023

@author: cameronthieme
"""

import math
import time
import csv

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

# finding correct row/column in grid
job = 0
xIter = job % N1
yIter = math.floor(job/N1) % N2

# theta value
th1 = xIter * (th1max - th1min) / N1 + (th1max - th1min) / (2 * N1) + th1min
th2 = yIter * (th2max - th2min) / N2 + (th2max - th2min) / (2 * N2) + th2min


# counting and labelling CM graphs
num_uniq = 0
label = 0

# name of true file
fname = 'MorseGraphs/MG_True_th1_' + str(th1) + '_th2_' + str(th2) 

# Info we want to save
infoList = [th1, th2, label]
# Naming File
fileNameResults = 'GridLabels/label_th1_' + str(th1)  + '_th2_' + str(th2) + '.csv'
# Writing label info to CSV
with open(fileNameResults, mode='w', newline='') as file:
    # Create a writer object
    writer = csv.writer(file)
    # Write the list to the CSV file
    writer.writerow(infoList)
# copying unique MG to other folder


# initializing list and dictionary of unique results
unique_mg_list = [fname]
unique_mg_dict = {fname:label}

for job in range(1,N1*N2):
    
    # finding correct row/column in grid
    xIter = job % N1
    yIter = math.floor(job/N1) % N2

    # theta value
    th1 = xIter * (th1max - th1min) / N1 + (th1max - th1min) / (2 * N1) + th1min
    th2 = yIter * (th2max - th2min) / N2 + (th2max - th2min) / (2 * N2) + th2min
    
    # name of true file
    fname = 'MorseGraphs/MG_True_th1_' + str(th1) + '_th2_' + str(th2) 
    
    matchCount = 0
    for filename in unique_mg_list:
        matchVal = ConleyMorseMatcher_FromFile(filename, fname)
        matchCount += matchVal
        if matchVal == 1:
            label = unique_mg_dict[filename]
            break
            
    if matchCount == 0:
        num_uniq += 1
        unique_mg_list.append(fname)
        unique_mg_dict[fname] = num_uniq
        label = num_uniq
        
    # Info we want to save
    infoList = [th1, th2, label]
    
    # Naming File
    fileNameResults = 'GridLabels/label_th1_' + str(th1)  + '_th2_' + str(th2) + '.csv'
    
    
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

