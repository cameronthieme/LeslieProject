import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

'''
Creates csv of true CM classes for grid; also jpg of same data

This file requires that trueGrid.py and gridLabeller.py are both done
also requires that 'GridLabels/trueLabelsDisarray.csv' has been made from terminal command

'''

myHeader = ['th1', 'th2', 'label']
df = pd.read_csv('GridLabels/trueLabelsDisarray.csv',  header = None)

df.columns = myHeader

th1List = sorted(list(df['th1'].unique()))
th2List = sorted(list(df['th2'].unique()))


df2 = pd.DataFrame(index=th2List,columns=th1List) 

for x,y in itertools.product(th1List, th2List):
#     print(x,y)
    df2[x][y] = df[(df['th1'] == x) & (df['th2'] == y)]['label'].item()
   
df2.to_csv('GridLabels/trueLabels.csv', index=True)


# Create a NumPy array with the same shape as the DataFrame
array = np.empty(df2.shape, dtype=float)

# Iterate over each element in the DataFrame
for row_idx, row in enumerate(df2.values):
    for col_idx, value in enumerate(row):
        array[row_idx, col_idx] = value
        

# Generate a random 25x40 array of integers between 0 and 13

# Create a color map for mapping the integers to colors
cmap = plt.get_cmap('tab20b')  # Adjust the colormap and number of colors as needed

# Plotting the array using imshow
plt.figure(figsize=(10, 8))  # Adjust the figure size as per your requirements
plt.imshow(array, cmap=cmap, aspect='auto')

# Setting up the plot boundaries and axis labels
plt.xlim(0, array.shape[1])
plt.ylim(0, array.shape[0])
# plt.xlabel('X', fontsize=12)
# plt.ylabel('Y', fontsize=12)
plt.title('Conley-Morse Classes')
# Remove x-axis and y-axis ticks
plt.xticks([])
plt.yticks([])

# Show the colorbar
cbar = plt.colorbar()
cbar.set_label('Conley-Morse Class', fontsize=10)  # Set the colorbar label

# Show the plot
plt.savefig('trueLabels.png')