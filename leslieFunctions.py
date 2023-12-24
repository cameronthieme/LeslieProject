#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:24:47 2023

Stores all functions that we need to run GP on Leslie Model

Default Setting is for large domain 
lower_bounds = [-0.001, -0.001]
upper_bounds = [320.056, 224.040]

@author: cameronthieme
"""


# Section 2: Importing packages and determining job

import CMGDB
# import matplotlib
import itertools
# import time
import math
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF#, WhiteKernel
# import csv
# import datetime
from interval import interval, imath
import networkx as nx
import cmgdb_utils
import graphviz


# Section 1: Regression Functions

# Define Leslie map (what we learn)
def leslie(x, th1 = 19.6, th2 = 23.68):
    return [(th1 * x[0] + th2 * x[1]) * math.exp (-0.1 * (x[0] + x[1])), 0.7 * x[0]]
    
def meas_err_traj_gen(noise_std = 0, traj_len = 1, th1 = 19.6, th2 = 23.68, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0]):
# Generate random point in the rectangle defined by lower_bounds, upper_bounds
    init = np.random.uniform(
                    low=[lower_bounds[0],lower_bounds[1]],
                    high=[upper_bounds[0], upper_bounds[1]], 
                    size=(1, 2))
    for images in range(traj_len): # will find traj_len consecutive images
        image = [leslie(init[-1], th1, th2)]
        init = np.append(init, image, axis = 0)
    # add noise to all images (measurement error)
    noise_meas = np.random.normal(loc=0.0,
                               scale=noise_std,
                               size=(len(init),2))
    init = init + noise_meas
    # add this trajectory to the data
    x_train = init[:-1]
    y_train = init[1:]
    return x_train, y_train

def step_err_traj_gen(noise_std = 0, traj_len = 1, th1 = 19.6, th2 = 23.68, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0]):
    '''

    Parameters
    ----------
    noise_std : TYPE, optional
        DESCRIPTION. The default is 0.
    traj_len : TYPE, optional
        DESCRIPTION. The default is 1.
    th1 : TYPE, optional
        DESCRIPTION. The default is 19.6.
    th2 : TYPE, optional
        DESCRIPTION. The default is 23.68.
    lower_bounds : TYPE, optional
        DESCRIPTION. The default is [-0.001, -0.001].
    upper_bounds : TYPE, optional
        DESCRIPTION. The default is [90.0, 70.0].

    Returns
    -------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    '''
    init = np.random.uniform(
        low=[lower_bounds[0],lower_bounds[1]],
        high=[upper_bounds[0], upper_bounds[1]], 
        size=(1, 2))
    for image_trial in range(traj_len):
        image = leslie(init[-1], th1, th2) + np.random.normal(loc=0.0,
                                   scale=noise_std,
                                   size=(1,2))
        for check in range(2):
            if image[0][check] < 0:
                image[0][check] = 0
        init = np.append(init, image, axis = 0)
    x_train = init[:-1]
    y_train = init[1:]
    return x_train, y_train

def embedder(arreh, delay_num = 1):
    traj = arreh[:,0]
    embedded = []
    for i in range(len(arreh) - delay_num):
        embedded.append([traj[i], traj[i + delay_num]])
    embedded = np.array(embedded)
    embed_input = embedded[:-1]
    embed_output = embedded[1:]
    return embed_input, embed_output

def sampGenLes(noise_std = 0, sample_size = 1000, noise_type = 'meas_err', traj_len = 1, th1 = 19.6, th2 = 23.68, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0], embed = False, delay_num = 1):
    '''
    Generates various samples of the Leslie model from the given domain.
    Origin appended to sample
    
    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of the Gaussian noise added to the sample. The default is 0.
    sample_size : int, optional
        Number of points in sample. The default is 1000.
    noise_type : string, optional
        How noise is added to the model.
        2 options:
            'meas_err':
                sample is taken, then error added to each output.  Simulutes a system that exactly obeys Leslie model, but is measured with imperfect equipment.
            'step_err':
                at each step in the trajectory, noise is added.  Simulates system which is governed by Leslie but also influenced by random factors, like Myvtan.
        The default is 'meas_err'.
    traj_len : int, optional
        Length of trajectories sampled. If 1, then randomly sampled from whole domain.  
        The default is 1.
    th1 : float, optional
        Parameter in Leslie Model. The default is 19.6.
    th2 : float, optional
        Parameter in Leslie Model. The default is 23.68.
    lower_bounds : list, optional
        Domain lower bounds.  [low0, low1]. The default is [-0.001, -0.001].
    upper_bounds : list, optional
        [high0, high1]. The default is [90.0, 70.0].

    Raises
    ------
    ValueError
        Error if sample size is not a multiple of trajectory length.

    Returns
    -------
    x_train : np array, Shape (sample_size, 2).
        input values of sample.  Shape (sample_size, 2).
    y_train0 : np array, shape (sample_size,)
        dim 0 of Leslie output.
    y_train1 : np array, shape (sample_size,)
        dim 1 of Leslie output.

    '''
    
    # initial warning to make sure that we can build the right sample size
    if sample_size % traj_len != 0:
        raise ValueError('Sample size must be a multiple of trajectory length')
    
    rand_inits = int(sample_size / traj_len) # number of total trajectories required
    
    if embed == True: # so that output embedded traj is original traj_len
        traj_len = traj_len + delay_num + 1 
    
    x_train = np.array([]).reshape(0,2)
    y_train = np.array([]).reshape(0,2)
    for rand_init in range(rand_inits):
        # build a single trajectory
        if noise_type == 'meas_err':
            x_traj, y_traj = meas_err_traj_gen(noise_std = noise_std, 
                                               traj_len = traj_len,
                                               th1 = th1,
                                               th2 = th2,
                                               lower_bounds = lower_bounds,
                                               upper_bounds = upper_bounds)
        elif noise_type == 'step_err':
            x_traj, y_traj = step_err_traj_gen(noise_std = noise_std, 
                                               traj_len = traj_len,
                                               th1 = th1,
                                               th2 = th2,
                                               lower_bounds = lower_bounds,
                                               upper_bounds = upper_bounds)
        # embedding step for single trajectory
        if embed == True:
            x_traj, y_traj = embedder(x_traj, delay_num)
        x_train = np.append(x_train, x_traj, axis = 0)
        y_train = np.append(y_train, y_traj, axis = 0)
    x_train = np.append(x_train, np.array([[0,0]]).reshape(1,2), axis = 0)
    y_train = np.append(y_train, np.array([[0,0]]).reshape(1,2), axis = 0)
    return x_train, y_train[:,0], y_train[:,1]

        
# define regressor
def GP(X_train, Y_train, tau = 6.445, beta = 1, noise_std = 0, n_restarts = 31):
    # fit Gaussian Process with dataset X_train, Y_train
    kernel = beta * RBF(length_scale = tau) #, length_scale_bounds = "fixed")
    # gp = GaussianProcessRegressor(kernel=kernel)
    # note that we must have 
    gp = GaussianProcessRegressor(kernel=kernel, alpha = max(1e-3,noise_std**2), n_restarts_optimizer=n_restarts)
    gp.fit(X_train, Y_train)
    return gp

def mu(X, gp0, gp1): # this will have same dimension as X
    '''
    input:
    X: list of 2 scalars
    gp0 & gp1: GPR models.  Input R^2, output R
    
    output:
    list len 2, each item scalar.  mean of each component function
    '''
    return [gp0.predict([X])[0], gp1.predict([X])[0]]

def sd(X, gp0, gp1): 
    '''
    input:
    X: list of 2 scalars
    gp0 & gp1: GPR models.  Input R^2, output R
    
    output:
    list len 2, each item scalar.  SD of each component function
    '''
    return [gp0.predict([X], return_std = True)[1][0], gp1.predict([X], return_std = True)[1][0]]

def zero_checker(gp0, gp1, check_number = 11, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0]):
    '''
    Check if mean map is trivial
    '''
    check_list = []
    for checks in range(check_number):
        rand_num = [random.uniform(lower_bounds[0],upper_bounds[0]),
                    random.uniform(lower_bounds[1], upper_bounds[1])]
        check_list.append((mu(rand_num, gp0, gp1)[0] == 0) or (mu(rand_num, gp0, gp1)[1]) == 0)
    if all(check_list):
        return 1
    else:
        return 0


# Section 3: Box Maps

# 3(a): Padded Box Map
def LeslieTrueBox(rect):
    return CMGDB.BoxMap(leslie, rect, padding=True)


# 3(b) GP Box Map
def CornerPoints(rect):
    '''
    Input rectangle: [low0, low1, high0, high1]
    Output: list of 4 corners of rectangle
    '''
    dim = int(len(rect) / 2)
    # Get list of intervals
    list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
    # Get points in the cartesian product of intervals
    X = [list(u) for u in itertools.product(*list_intvals)]
    return X

def BoxMapSD(rect, gp0, gp1):
    '''
    Find image of rect using mean map
        corners and center mapped
    Padded by sd
    fed gp0 & gp1 since those define mu, sd
    '''
    X = CornerPoints(rect) # points we want to evaluate at
    center = [(rect[2] + rect[0])/2, (rect[3] + rect[1])/2]
    X.append(center)
    sd_center = sd(center, gp0, gp1)
    Y = [mu(x, gp0, gp1) for x in X] # image of the corner points under mean map
    Y_l_bounds = [min([y[d] for y in Y]) - sd_center[d] for d in range(2)]
    Y_u_bounds = [max([y[d] for y in Y]) + sd_center[d] for d in range(2)]
    f_rect = Y_l_bounds + Y_u_bounds
    return f_rect 

# 3(c): Interval Box Map

def leslieInterval(x, th1 = 19.6, th2 = 23.68):
    th1 = interval[th1]
    th2 = interval[th2]
    return [(th1 * x[0] + th2 * x[1]) * imath.exp (-interval[0.1] * (x[0] + x[1])),
            interval[0.7] * x[0]]

def IntervalBoxMap2D(f, rect, th1 = 19.6, th2 = 23.68):
    # Get endpoints defining rect
    x1, y1, x2, y2 = rect
    # Define interval box x
    x = [interval[x1, x2], interval[y1, y2]]
    # Evaluate f as an interval map
    y = f(x, th1, th2)
    # Get endpoints of y
    # y[0] is the first variable interval
    # y[1] is the second variable interval
    x1, x2 = y[0][0].inf, y[0][0].sup
    y1, y2 = y[1][0].inf, y[1][0].sup
    return [x1, y1, x2, y2]

# def leslieInterval(x):
#     th1 = interval[19.6]
#     th2 = interval[23.68]
#     return [(th1 * x[0] + th2 * x[1]) * imath.exp (-interval[0.1] * (x[0] + x[1])),
#             interval[0.7] * x[0]]

# def IntervalBoxMap2D(f, rect):
#     # Get endpoints defining rect
#     x1, y1, x2, y2 = rect
#     # Define interval box x
#     x = [interval[x1, x2], interval[y1, y2]]
#     # Evaluate f as an interval map
#     y = f(x)
#     # Get endpoints of y
#     # y[0] is the first variable interval
#     # y[1] is the second variable interval
#     x1, x2 = y[0][0].inf, y[0][0].sup
#     y1, y2 = y[1][0].inf, y[1][0].sup
#     return [x1, y1, x2, y2]

# def LeslieIntervalBox(rect):
#     return IntervalBoxMap2D(leslieInterval, rect)

# Section 4: Matching Morse Graphs

def ConleyMorseMatcher(morse_graph_1, morse_graph_2):
    '''
    import: 2 conley morse graphs
    output: Binary, whether the nontrivial components of Morse Graph match
    '''
    
    # Getting Nontrivial Parts
    nontrivial_mg_1 = cmgdb_utils.NonTrivialCMGraph(morse_graph_1)
    nontrivial_mg_2 = cmgdb_utils.NonTrivialCMGraph(morse_graph_2)
    
    # 7(a): Graph Isomorphism
    # initializing nx graphs
    G1 = nx.Graph()
    G2 = nx.Graph()
    
    # Add nodes and edges to the graphs
    G1.add_nodes_from(nontrivial_mg_1.vertices())
    G1.add_edges_from(nontrivial_mg_1.edges())
    G2.add_nodes_from(nontrivial_mg_2.vertices())
    G2.add_edges_from(nontrivial_mg_2.edges())
    
    # truth statement for if the graphs are isomorphic
    isomorphism = nx.is_isomorphic(G1, G2)
    
    # 7(b): Labels Identical
    # checking if labels are identical
    labels1 = {nontrivial_mg_1.vertex_label(vert) for vert in nontrivial_mg_1.vertices()}
    labels2 = {nontrivial_mg_2.vertex_label(vert) for vert in nontrivial_mg_2.vertices()}
    labelsSame = (labels1 == labels2)
    
    # 7(c): Edges map to properly labelled edges
    
    # creating node to label dictionary:
    verts1 = [vert for vert in nontrivial_mg_1.vertices()]
    vertLabels1 = [nontrivial_mg_1.vertex_label(vert) for vert in nontrivial_mg_1.vertices()]
    dict1 = dict(zip(verts1, vertLabels1))
    
    verts2 = [vert for vert in nontrivial_mg_2.vertices()]
    vertLabels2 = [nontrivial_mg_2.vertex_label(vert) for vert in nontrivial_mg_2.vertices()]
    dict2 = dict(zip(verts2, vertLabels2))
    
    # List of edges, given as labels
    edgeLabels1 = []
    edgeLabels2 = []
    for edge in nontrivial_mg_1.edges():
        edgeLabels1.append((dict1[edge[0]], dict1[edge[1]]))
    for edge in nontrivial_mg_2.edges():
        edgeLabels2.append((dict2[edge[0]], dict2[edge[1]]))
        
    # matching these edge labels
    for perm2 in itertools.permutations(edgeLabels2):
        if edgeLabels1 == list(perm2):
            edgeLabelMatch = True
            break
        edgeLabelMatch = False
    
    # Section 7: Saving Results
    if isomorphism & labelsSame & edgeLabelMatch:
        return 1
    else:
        return 0 

def ConleyMorseMatcher_FromFile(fname1, fname2):
    '''
    import: 2 filenames corresponding to dot files
        Dot files are conley morse graphs
    output: Binary, whether the nontrivial components of Morse Graph match
    '''
    
    morse_graph_1 = G1 = cmgdb_utils.graph_from_dotfile(fname1)
    morse_graph_2 = G2 = cmgdb_utils.graph_from_dotfile(fname2)
    
    # Getting Nontrivial Parts
    nontrivial_mg_1 = cmgdb_utils.NonTrivialCMGraphPyChomP(morse_graph_1)
    nontrivial_mg_2 = cmgdb_utils.NonTrivialCMGraphPyChomP(morse_graph_2)
    
    # 7(a): Graph Isomorphism
    # initializing nx graphs
    G1 = nx.Graph()
    G2 = nx.Graph()
    
    # Add nodes and edges to the graphs
    G1.add_nodes_from(nontrivial_mg_1.vertices())
    G1.add_edges_from(nontrivial_mg_1.edges())
    G2.add_nodes_from(nontrivial_mg_2.vertices())
    G2.add_edges_from(nontrivial_mg_2.edges())
    
    # truth statement for if the graphs are isomorphic
    isomorphism = nx.is_isomorphic(G1, G2)
    
    # 7(b): Labels Identical
    # checking if labels are identical
    labels1 = {nontrivial_mg_1.vertex_label(vert) for vert in nontrivial_mg_1.vertices()}
    labels2 = {nontrivial_mg_2.vertex_label(vert) for vert in nontrivial_mg_2.vertices()}
    labelsSame = (labels1 == labels2)
    
    # 7(c): Edges map to properly labelled edges
    
    # creating node to label dictionary:
    verts1 = [vert for vert in nontrivial_mg_1.vertices()]
    vertLabels1 = [nontrivial_mg_1.vertex_label(vert) for vert in nontrivial_mg_1.vertices()]
    dict1 = dict(zip(verts1, vertLabels1))
    
    verts2 = [vert for vert in nontrivial_mg_2.vertices()]
    vertLabels2 = [nontrivial_mg_2.vertex_label(vert) for vert in nontrivial_mg_2.vertices()]
    dict2 = dict(zip(verts2, vertLabels2))
    
    # List of edges, given as labels
    edgeLabels1 = []
    edgeLabels2 = []
    for edge in nontrivial_mg_1.edges():
        edgeLabels1.append((dict1[edge[0]], dict1[edge[1]]))
    for edge in nontrivial_mg_2.edges():
        edgeLabels2.append((dict2[edge[0]], dict2[edge[1]]))
        
    # matching these edge labels
    for perm2 in itertools.permutations(edgeLabels2):
        if edgeLabels1 == list(perm2):
            edgeLabelMatch = True
            break
        edgeLabelMatch = False
    
    # Section 7: Saving Results
    if isomorphism & labelsSame & edgeLabelMatch:
        return 1
    else:
        return 0 
   
# saving files for reduced CM graphs
def ReducedGraphSaver(oldFile, newFile):
    '''
    Saves the reduced CM graph of a CM graph file
    Input: 
        morseFileName -- string, location/name of original graph file
        saveName -- string, location/name of reduced graph file
    '''
    morse_graph_unreduced = cmgdb_utils.graph_from_dotfile(oldFile)
    nontriv_mg = cmgdb_utils.NonTrivialCMGraphPyChomP(morse_graph_unreduced)
    nontriv_mg_plot = nontriv_mg.graphviz()
    graphviz.Source(nontriv_mg_plot).save(newFile)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    