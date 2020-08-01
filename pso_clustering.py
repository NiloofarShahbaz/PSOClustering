from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from particle import Particle


class PSOClusteringSwarm:
    def __init__(self, n_clusters: int, n_particles: int, data: np.ndarray, hybrid=True, w=0.72, c1=1.49, c2=1.49):
        """
        Initializes the swarm.
        :param n_clusters: number of clusters
        :param n_particles: number of particles
        :param data: ( number_of_points x dimensions)
        :param hybrid: bool : whether or not use kmeans as seeding
        :param w:
        :param c1:
        :param c2:
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data

        self.particles = []
        # for storing global best
        self.gb_pos = None
        self.gb_val = np.inf
        # global best data clustering so far
        # for each data point will contain the cluster number
        self.gb_clustering = None

        self._generate_particles(hybrid, w, c1, c2)

    def _print_initial(self, iteration, plot):
        print('*** Initialing swarm with', self.n_particles, 'PARTICLES, ', self.n_clusters, 'CLUSTERS with', iteration,
              'MAX ITERATIONS and with PLOT =', plot, '***')
        print('Data=', self.data.shape[0], 'points in', self.data.shape[1], 'dimensions')

    def _generate_particles(self, hybrid: bool, w: float, c1: float, c2: float):
        """
        Generates particles with k clusters and t-dimensional points
        :return:
        """
        for i in range(self.n_particles):
            particle = Particle(n_clusters=self.n_clusters, data=self.data, use_kmeans=hybrid, w=w, c1=c1, c2=c2)
            self.particles.append(particle)

    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.pb_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        """

        :param plot: = True will plot the global best data clusters
        :param iteration: number of max iteration
        :return: (best cluster, best fitness value)
        """
        self._print_initial(iteration, plot)
        progress = []
        # Iterate until the max iteration
        for i in range(iteration):
            if i % 200 == 0:
                clusters = self.gb_clustering
                print('iteration', i, 'GB =', self.gb_val)
                print('best clusters so far = ', clusters)
                if plot:
                    centroids = self.gb_pos
                    if clusters is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=clusters, cmap='viridis')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:  # if there is no clusters yet ( iteration = 0 ) plot the data with no clusters
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_pb(data=self.data)
                self.update_gb(particle=particle)

            for particle in self.particles:
                particle.move_centroids(gb_pos=self.gb_pos)
            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])

        print('Finished!')
        return self.gb_clustering, self.gb_val
