# LeslieProject

Code for the paper using CMGDB on the Leslie model

## Creating the Noise/Sample Size Table

One main goal of this repository is to efficiently construct the noise/sample size tables from the paper.  This requires extensive computational time, and our computations were performed on Rutgers's Amarel Cluster.  

To prepare the code for use on the Amarel cluster, first ensure that all packages are installed from each file.  Then, create a folder with the files
tableMaker400.py
leslieFunctions.py
Runner.sh
Builder.sh
buildTable.sh

Additionally, create three blank folders in that folder: MorseGraphs, OutFiles, ResultTables.  These folders will store the information about the trial runs (we run thousands of trials per batch so it's best to keep the area organized).  

In tableMaker400.py, select the type of experiment you would like to run by changing a few variables.  Namely, set desired values for
traj_len (length of individual trajectories that make up our sample)
noise_type (either 'meas_err' or 'step_err')
embed (True for an embedded 1d trajectory, False for a full 2d trajectory.

Run the command 
sbatch Runner.sh

