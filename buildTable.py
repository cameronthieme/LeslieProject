# -*- coding: utf-8 -*-
"""
Sep 25

This script takes in the large csv file containing the success/failure of each test
It outputs a table with that information organized

First must sbatch Runner.sh, which calls tableMaker400.py

After each of those jobs finishes (400 jobs total), call
cat res* superFile.csv
in the Results folder

Then this file can be run, which creates the final, human-readable table
"""




import pandas as pd
import itertools

myHeader = ['train_size', 'traj_len', 'noise_std', 'noise_type', 'embed', 'identical', 'zero_counter']

df = pd.read_csv('ResultTables/superFile.csv',  header = None)

df.columns = myHeader

trainSizeList = [100, 200, 500, 1000, 2000]

noiseList = [0, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2]

df2 = pd.DataFrame(index=trainSizeList,columns=noiseList) 
df3 = pd.DataFrame(index=trainSizeList,columns=noiseList) 

for x,y in itertools.product(trainSizeList, noiseList):
    bistab_detected = df.loc[(df['train_size'] == x) & (df['noise_std'] == y), 'identical'].sum()
    zero_detected = df.loc[(df['train_size'] == x) & (df['noise_std'] == y), 'zero_counter'].sum()
    df2[y][x] = (bistab_detected, zero_detected)
    df3[y][x] = bistab_detected

df2.to_csv('finalTable_withZeros.csv', index=True)
df3.to_csv('finalTable.csv', index=True)
print(df2)
print(df3)