

class particle:

    def __init__(self):
        #initialize random weights and velocities
        self.position = [] # numpy array, == weights
        self.velocity = [] 
        self.fitness = 5
        self.personal_best = 3


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