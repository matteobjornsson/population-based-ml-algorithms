import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import matplotlib.pyplot as plt



class individual(): 
 
    def __init__(self, Size):
            self.fitness = float('inf')
            self.Chromie = list() 
            lowerbound = -10 
            upperbound = 10 
            for i in range(Size):
                Chromie.append(random.uniform(lowerbound,upperbound))
            self.Chromie = np.array(self.Chromie)
    def getchromie(): 
        return self.Chromie
    def setchromie(chrom):
        self.Chromie = chrom 
    def getfit(): 
        return self.fitness
    def setfit(fitt): 
        self.fitness = fitt

class DE:
    #this is the hyperparameter from mutation

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list,Chromie_Size,nn):
         #hyperparameter?
        self.pop_size = 10
        self.population = list() #TODO: what data structure to use here?
        for i in range(self.pop_size): 
            temp = individual(Chromie_Size)
            self.population.append(temp)
        self.generation = 0
        # hyperparameter!
        self.probability_of_crossover = 75
        #hyperparameter
        self.beta = .2
        self.nn = nn 
        self.maxgens = 1000 
        self.globalbest = list() 

    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def fitness(self,chromie) -> float:
        return self.nn.fitness(chromie)


    ###################################
    # grab 3 vectors from pop, without repalcement, generate trial vector
    ###############################
    def mutate_and_crossover(self):
        bestfit = float('inf')
        for i in range(len(self.population)):
            nums = list()
            nums.append(i)
            count = 0 
            while(count != 3): 
                org = random.randint(0,len(self.population))
                if org in nums: 
                    continue 
                nums.append(org)
                count = count +1
            organism = self.population[i]
            nums.remove(i)
            organ1 = self.population[nums[0]]
            organ2 = self.population[nums[1]]
            organ3 = self.population[nums[2]]
            temp = organmism.getchromie()
            for j in range(len(organism.getchromie())): 
                ColumnTV= organ1.getchromie()[j] + self.beta * ((organ2.getchromie()[j] * organ2.getchromie()[j]) - (organ3.getchromie()[j] * organ3.getchromie()[j]))      
                coin = random.randint(0,99) + 1 
                if coin > self.probability_of_crossover: 
                    temp[j] = ColumnTV
                else: 
                    #No crossover 
                    continue 
            fitness = self.fitness(temp)
            if fitness < organmism.getfit(): 
                organism.setfit(fitness)
                organism.setchromie(temp)
            if fitness < bestfit: 
                bestfit = finess
            
            self.population[i] = organism
        self.globalbest.append(bestfit)

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
