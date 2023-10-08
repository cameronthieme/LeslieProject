# LeslieProject

Code for the paper using CMGDB on the Leslie model.

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
