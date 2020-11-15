
class individual:
    
    def __init__(self):
        self.fitness = 10
        #How big should each chromosome be? My initial assumption is the number of feature vectors across the board 
        self.chromosome = [] # some numpy == weights


    def SetChromie(Feature_Size):
        #Loop through each index up until the number of features and just set it to 0 
        for i in range(Feature_Size): 
            self.chromosome[i] = 0  

    def ReturnChromie():
        return self.chromosome

    def printChromie(): 
        for i in self.chromosome: 
            print(i)

class GA:
    #hyperparameter?
    pop_size = 10
    population = [] #TODO: what data structure to use here?
    genetation = 0
    Chromosome_Size = 10 
    # hyperparameter!
    probability_of_crossover = .5 

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list):
        #init general population 
        #On the creation of a genetic algorithm, we should create a series of random weights in a numpy array that can be fed into the neural network. 
        #Create a list to hold all of the individual objects and set each one with random weights to then send through the algorithm 
        Population = list() 
        #Create an individual object and set the chromosome weight randomly for each of the individuals in the population (pop size)
        for i in range(pop_size): 
            #Create a new individual object 
            temp = self.individual(Chromosome_Size)
            
            
            #Add the individual to the list of total population 
            Population.append(temp)

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

##################################
# Main function down here? 
# Remember, functions we can dispatch as jobs with unique parameters = parallelizeable. 
#################################
    def driver(input_parameters): 
        # Until convergence: 
            # select individuals from pop to mate
            # crossover
            # mutate
            # evaluate fitness of new individuals
            # replace existing population
        pass



if __name__ == '__main__':
    print("Program Start")






    print("Program End ")
