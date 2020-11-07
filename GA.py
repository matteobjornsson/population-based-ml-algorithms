

class GA:
    #hyperparameter?
    pop_size = 10
    population = [] #TODO: what data structure to use here?
    genetation = 0
    # hyperparameter!
    probability_of_crossover = .5 

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

    ##################################
    # pick a subset of POP based on fitness OR some sort of random or ranked selection
    #####################################
    def selection(self):
        #TODO: what does this function return? Does it modify population in place? does it return a subselection of pop? 
        # is that subselection a list of individuals or just like array indices?

        # TODO: pick the selection mechanism
            # proportionate, rank, or tournament
            # use rank? would work well with a max priority queue/heap
        pass

    ####################################
    # make new generation based on parent selection by swapping chromosomes 
    ####################################
    def crossover(self, set_of_all_selected_parents): 
        #TODO: pick crossover mechanism (uniform?)
        #TODO: how many individuals are we generating? Entirely new population? 
        pass

    ###################################
    # introduce random change to each individual in the generation
    ###############################
    def mutate(self, new_generation):
        #TODO: pick mutation mechanism.
        pass


# Until convergence: 
    # select individuals from pop to mate
    # crossover
    # mutate
    # evaluate fitness of new individuals
    # replace existing population