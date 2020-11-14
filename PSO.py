
import numpy as np

class Particle:

    def __init__(self, position_range: float, velocity_range: float, input_size: int, layers: list, output_size = 1):

        total_layers = [input_size] + layers + [output_size]
        total_particles = 0
        for i in range(len(total_layers)-1):
            total_particles += total_layers[i] * total_layers[i+1]
        self.position = np.random.uniform(-position_range, position_range, total_particles)
        self.velocity = np.random.uniform(-velocity_range, velocity_range, total_particles)
        self.fitness = float('inf')
        self.pbest_position = self.position
        self.pbest_fitness = self.fitness


class PSO:

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list):
        #init general population 
            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN

        # INIT BOTH *POSITION* (WEIGHTS) AND *VELOCITY*

        #TODO: figure out if we should init a NN here or pass in, etc. What layer does this file represent
        self.pop_size = 10
        self.population= [] 
        self.global_best = 10
        self.t = 0
        self.max_t = 1000
        #HYPERPARAMETERS:
        self.omega = .5
        self.c1 = .3
        self.c2 = .3
        self.vmax = 10

    def swarm_diversity(self) -> float:
        pass

    def swarm_fitness(self) -> float:
        # plot this over time to determine convergence?
        pass

    def update_position_and_velocity(self):
        # iterate over each particle
            # update v and x using equations from class
        pass

    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def fitness(self, individual) -> float:
        # this applies the individual's weights to the NN, feeds data set through and returns error of forward pass
        # TODO: figure out if we pass through entire dataset, or batch, etc. 
        pass

    ####################################
    # driver method
    ####################################
    # initialize
    # until convergence of global best:
        # update V and X for each swarm memeber and eval fitness

if __name__ == '__main__':
    p = Particle(10, 1, 7, [7,2], 1)
