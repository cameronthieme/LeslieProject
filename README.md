# LeslieProject

This repository features all of the code for the paper using CMGDB on the Leslie 
population model (work in progress, no title yet). The general aim of the paper is to give some 
computational estimates of the long term reliability of dynamical systems surrogate models. 

To understand this work, first note that the qualitative behavior of a dynamical system can 
be described using something called a Conley-Morse graph (CMG).  This tool is pretty 
complicated, but we should mention that it can rigorously detect things like bistability in a 
system. CMGDB is a software that determines the CMG of a given system.

We analyze the Leslie population model and compare the qualitative behavior of Gaussian process 
(GP) surrogate models with the true dynamics of the system; this qualitative behavior is 
described via the CMG.  We perform this analysis at various parameter values, noise levels, and 
sample sizes in order to obtain some idea of how the surrogate models perform across these 
dimensions. The basic steps, for a given parameter vector theta, noise level sigma, and sample 
size N, are to
* randomly sample N points in our domain and find their (noisy) images under the Leslie map
* Create a GP surrogate model using this sample and noise level sigma
* Compare the CMG of the GP surrogate model with the CMG of the true Leslie model at that 
parameter value theta
* Repeat these steps 50 times and report the number of successes

While fixing a single parameter vector and varying noise levels and sample sizes we create tables 
that give an idea of how successful surrogate modeling can be across these factors.  Fixing 
a noise level and sample size and varying the parameter vector (2 parameters), we create a 
heatmap to convey the same. This software allows us to do both of these things.  

There's a lot more that goes into this process; we dabble in various sampling methodologies and 
time series delay embedding.  To get a real idea of what's going on you'll have to read the 
paper (whenever we publish it).

Note: README file currently only contains instructions for Table creation, not heatmap

## Creating the Noise/Sample Size Table

One main goal of this repository is to efficiently construct the noise/sample size tables from the paper.  This requires extensive computational time, and our computations were performed on Rutgers's Amarel Cluster.  

To prepare the code for use on the Amarel cluster, first ensure that all packages are installed from each file.  Then, create a folder with the following files:<br>
tableMaker400.py <br>
leslieFunctions.py <br>
Runner.sh <br>
Builder.sh <br>
buildTable.py <br>
The name "tableMaker400" refers to the fact that this file is called 400 times (i.e. simultaneously on 400 nodes of the cluster) in order to make the table.

Additionally, create three blank folders in that folder: MorseGraphs, OutFiles, ResultTables.  These folders will store the information about the trial runs (we run thousands of trials per batch so it's best to keep the area organized).  

In tableMaker400.py, select the type of experiment you would like to run by changing a few variables.  Namely, set desired values for <br>
traj_len (length of individual trajectories that make up our sample) <br>
noise_type (either 'meas_err' or 'step_err') <br>
embed (True for an embedded 1d trajectory, False for a full 2d trajectory.

Run the command  <br>
sbatch Runner.sh <br>
in order to run the computations necessary for the table.  This will place 2000 .csv files in the ResultTables folder.  The reason we have 2000 files is that our tables are 8x5 (40 entries), and each entry lists the outcome of 50 experiments at the given noise/sample size value (50*40=2000).  

When all of these computations are finished, go into the ResultTables folder, and run the operation <br>
cat r* > superFile.csv <br>
This will create one .csv file which contains the results of all 2000 experiments.  When this is done, navigate back up one level to the original folder, and run the command <br>
sbatch Builder.sh <br>
This will result in two tables: <br>
finalTable_withZeros.csv <br>
finalTable.csv <br>
The first file (withZeros) contains a tuple in each of the table entries.  The first entry in the tuple is the number of successes (out of 50 possible) in detecting the correct dynamics.  The second entry gives the number of times that the GP gave a numerical error and gave the zero map.  Because we used 40 restarts for the GP each time, each of the second entries should be zero (i.e. the GP never fails to create a surrogate model).  If this is the case, then the second table (finalTable.csv) gives the desired table in .csv form.
