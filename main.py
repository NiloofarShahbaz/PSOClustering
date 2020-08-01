import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm

plot = True
data_points = pd.read_csv('iris.txt', sep=',', header=None)
clusters = data_points[4].values
data_points = data_points.drop([4], axis=1)
# if you want to plot you can only show 2 points! so will use 2 points of data
if plot:
    data_points = data_points[[0, 1]]
# convert to numpy 2d array
data_points = data_points.values
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)

# For showing the actual clusters
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
print('Actual classes = ', clusters)
