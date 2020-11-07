

class DE:
    #hyperparameter?
    pop_size = 10
    population = [] #TODO: what data structure to use here?
    genetation = 0
    # hyperparameter!
    probability_of_crossover = .5 
    #hyperparameter
    beta = .2 #this is the hyperparameter from mutation

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list):
        #init general population 
            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN

        #TODO: figure out if we should init a NN here or pass in, etc. What layer does this file represent
        pass

    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def fitness(self, individual) -> float:
        # this applies the individual's weights to the NN, feeds data set through and returns error of forward pass
        # TODO: figure out if we pass through entire dataset, or batch, etc. 
        pass

    ###################################
    # grab 3 vectors from pop, without repalcement, generate trial vector
    ###############################
    def mutate_and_crossover(self):
        #for each pop individual:
            # select three vectors x1 x2 x3 from pop
            # calculate trial vector uj = x1 + beta * (x2 - x3)
            # generate new vector via crossover per feature in individual
            # evaluate fitness of new vector
                # keep the one with the better fitness
            # TODO: do we edit each individual in place, from i=0 to len(pop)?
        pass

    ##################################
    # Main function down here? 
    # Remember, functions we can dispatch as jobs with unique parameters = parallelizeable. 
    #################################
    def driver(input_parameters): 
        # Until convergence: 
            # for each individual:
                # mutate and crossover to generate replacement 
                # eval fitness of replacement, keep better of the two
        pass
