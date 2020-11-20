import numpy as np
import math
import random 


class individual:
    
    def __init__(self):
        self.fitness = 10
        #How big should each chromosome be? My initial assumption is the number of feature vectors across the board 
        self.chromosome = [] # some numpy == weights
        self.Size = 0 


    def InitChromie(Feature_Size):
        #Loop through each index up until the number of features and just set it to 0 
        for i in range(Feature_Size): 
            self.chromosome[i] = 0  

    def SetChromie(Chromos): 
        self.chromosome = Chromos
    def SetSize(si): 
        self.Size = si 
    def getChromie(): 
        return self.chromosome 

    def ReturnChromie():
        return self.chromosome

    def printChromie(): 
        for i in self.chromosome: 
            print(i)

class GA:
    #hyperparameter?
    pop_size = 10
    population = list() #TODO: what data structure to use here?
    genetation = 0
    Chromosome_Size = 10 
    # hyperparameter!
    probability_of_crossover = .5 

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list, LayerCount,Feature_Size,NN):

        self.Layer_Node_Count = LayerCount
        #SEt the size to be the number of features 
        self.Chromosome_Size = Feature_Size 
        #Take in a neural Network 
        self.nn = NN 
        self.fit = list() 
        #init general population 
        #On the creation of a genetic algorithm, we should create a series of random weights in a numpy array that can be fed into the neural network. 
        #Create an individual object and set the chromosome weight randomly for each of the individuals in the population (pop size)
        for i in range(pop_size): 
            #Create a new individual object 
            temp = self.individual()
            #Set the array size 
            temp.SetSize(Feature_Size)
            #Initialize an empty list of weights 0s
            temp.InitChromie(Chromosome_Size)
            #Now randomly generate values to start for each of these sizes 
            temp.setChromie(self.GenerateWeights())
            #Add the individual to the list of total population 
            self.population.append(temp)

            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN


    #Generating the initial weights 
    def GenerateWeights(): 
        # initialize weights randomly, close to 0
        # generate the matrices that hold the input weights for each layer. Maybe return a list of matrices?
        # will need 1 weight matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        weights = []
        counts = self.layer_node_count
        for i in range(self.layers):
            if i == 0:
                weights.append([])
            else:
                # initialze a (notes, inputs) dimension matrix for each layer. 
                # layer designated by order of append (position in weights list)
                layer_nodes = counts[i]
                layer_inputs = counts[i-1]
                weights.append(np.random.randn(layer_nodes, layer_inputs) * 1/layer_inputs) # or * 0.01
        return weights



    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def fitness(self, individual) -> float:
        #Fitness Function will be Mean Absolute Error
        for i in self.population:  
            fitscore = self.nn.fitness(i.getChromie()) 
            self.fit.append(fitscore)
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
        newPopulation = list()
        newFitness = list()  
        Subset = 0 
        Subset = self.pop_size / 2 
        if Subset % 2 == 1: 
            Subset = Subset + 1 
        for i in range(Subset): 
            mins = 0
            #Find the minimum subset times 
            for i in range(len(self.fit)):
                if self.fit[mins] < self.fit[i]: 
                    mins = i 
                continue 
            newFitness.append(self.fit[mins])
            self.fit.remove(self.fit[mins])
            newPopulation.append(self.population[mins])
            self.population.remove(self.population[mins])
        self.population = newPopulation
        self.fit = newFitness

    ####################################
    # make new generation based on parent selection by swapping chromosomes 
    ####################################
    def crossover(self, set_of_all_selected_parents): 
        #TODO: pick crossover mechanism (uniform?)


        while(len(self.population) > self.pop_size): 
            Kill = random.randint(0,len(self.population))
            self.population.remove(self.population[Kill])

    ###################################
    # introduce random change to each individual in the generation
    ###############################
    def mutate(self, new_generation):
        for i in self.population: 
            perc = random.randint(0,99) + 1 
            if perc < 85: 
                continue 
            else: 
                #What to mutate and to what? 

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
