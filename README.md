# PSOClustering
**This is an implementation of clustering data with particle swarm optimization(PSO) algorithm**

This implementation is inspired by the following paper : [Data Clustering using Particle Swarm Optimization](https://ieeexplore.ieee.org/document/1299577)

The dataset used in this implementation is the [IRIS flower dataset](https://archive.ics.uci.edu/ml/datasets/iris) but this can surely work with other datasets too!

### Run
after installing the requirements run *main.py*. you can choose to plot the point with matplotlib by changing the variable `plot` at the top of the code. 

However **be aware that this will only keep the first 2 dimensions of the dataset points** and other dimensions will not be considered.

Other than the number of clusters (`n_clusters`) and the number of particles (`n_particles`) you can choose whether to use kmeans for seeding the initial swarm( called Hybrid PSO Clustering) or not ( with `hybrid` variable). You can also change the pso algorithm parameters `w`, `c1` and `c2`.

### Usage
will need *pso_clustering.py* and *particle.py*.

`from pso_clustering import PSOClusteringSwarm`

`# data should be a numpy array of n-dimensional points`

`pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True, w=0.72, c1=1.49, c2=1.49)`

`clusters, global_best_fitness = pso.start(iteration=1000)`

The function *start()* will return a tuple of the final clusters (for each data point has the cluster id) and the final value of the global best fitness of the swarm. 
