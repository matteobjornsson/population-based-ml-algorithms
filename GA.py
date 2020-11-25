
# Author: Nick Stone, some by Matteo Bjornsson
################################################################################
# This is the driver for running the GA experiment. This file 
# grabs data sets from "./NormalizedData" and for each data set generates 
# tenfolds, and runs GA to train each of the 0,1,2 layer neural networks. 
################################################################################
import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import matplotlib.pyplot as plt

#Individual that is designed to represent a given member in the population 
class individual:
    #On init set the meta data to some known values 
    def __init__(self):
        self.fitness = float('inf')
        #How big should each chromosome be? My initial assumption is the number of feature vectors across the board 
        self.chromosome = [] # some numpy == weights
        self.Size = 0 

    #Use a separate init function when given some meta data 
    def InitChromie(self,Feature_Size):
        #Loop through each index up until the number of features and just set it to 0 
        self.chromosome = [Feature_Size]
        for i in range(len(self.chromosome)): 
            self.chromosome[i] = 0  
        self.chromosome = np.array(self.chromosome)
        self.Size = Feature_Size
    ###################################### Mutators and Accessors ##########################
    #SEt the fitness of a member in the population 
    def setfit(self,fit): 
        self.fitness = fit 
    #Get the members fitness 
    def getfit(self): 
        return self.fitness 
    #Set the members chromosome 
    def SetChromie(self,Chromos): 
        self.chromosome = Chromos
    #Set the members size 
    def SetSize(self,si): 
        self.Size = si 
    #Get the chromosome size 
    def getsize(self): 
        return self.Size
    #Get the chromosome from a member in the population 
    def getChromie(self): 
        return self.chromosome 
    #Return the given member in the populations chromosome 
    def ReturnChromie(self):
        return self.chromosome
    #Print the contents of the chromosome 
    def printChromie(self): 
        for i in self.chromosome: 
            print(i)

class GA:

    #####################
    # Initialize the population etc
    ####################
    #
    def __init__(self, hyperparameters:dict , Total_Weight:int ,NN):
        
        self.maxGen = hyperparameters["maxGen"]
        self.pop_size = hyperparameters["pop_size"]
        self.mutation_rate = hyperparameters["mutation_rate"]
        self.mutation_range = hyperparameters["mutation_range"]
        self.crossover_rate = hyperparameters["crossover_rate"]
        self.generation = 0 
        #SEt the size to be the number of features 
        self.Chromosome_Size = Total_Weight
        #Take in a neural Network 
        self.nn = NN 
        self.globalfit = list() 
        
        #init general population 
        #Create an individual object and set the chromosome weight randomly for each of the individuals in the population (pop size)
        self.population = list()
        for i in range(self.pop_size): 
            #Create a new individual object 
            temp = individual()
            #Set the array size 
            temp.SetSize(Total_Weight)
            #Initialize an empty list of weights 0s
            temp.InitChromie(Total_Weight)
            #Now randomly generate values to start for each of these sizes 
            temp.SetChromie(self.GenerateWeights())
            #Add the individual to the list of total population 
            self.population.append(temp)

            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN
        self.bestChromie = self.population[0]


    #Generating the initial weights 
    def GenerateWeights(self): 
        # initialize weights randomly, close to 0
        # generate the matrices that hold the input weights for each layer. Maybe return a list of matrices?
        # will need 1 weight matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        layer_nodes = - 1
        layer_inputs = 1 
        weights = np.random.uniform(layer_nodes, layer_inputs,self.Chromosome_Size)
        return weights



    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def fitness(self,) -> float:
        #Fitness Function will be Mean squared Error
        for i in self.population:  
            fitscore = self.nn.fitness(i.getChromie()) 
            i.setfit(fitscore)
    
        ########################################
    # Evaluate the fitness of an individual
     # USED IN VIDEO TO PRINT EACH STEP DO NOT USE IN PROD 
    ########################################
    def pfitness(self,) -> float:
        print("FITNESS")
        #Fitness Function will be Mean squared Error
        for i in self.population:  
            fitscore = self.nn.fitness(i.getChromie()) 
            print(fitscore)
            i.setfit(fitscore)
            

    ##################################
    # pick a subset of POP based on fitness OR some sort of random or ranked selection
    #####################################
    def selection(self):
        #Sort the population based on the fitness function 
        self.population = sorted(self.population, key=lambda individual: individual.fitness)
        #Set a variable to the most fit member 
        bestChromie = self.population[0]
        #add this member to the global most fit 
        self.globalfit.append(bestChromie.fitness)
        #If this new mem is the most for the all of the generations
        if bestChromie.fitness < self.bestChromie.fitness:
            #Update the best chromosome 
            self.bestChromie = bestChromie
        #Set the pop size to a local variable 
        pop = self.pop_size

        #  RANKED ROULETTE SELECTION
        newPopulation = list()
        #set a counter for half of the population + 1 to cross over 
        Subset = int(pop / 2 )
        Subset = Subset + 1 
        for j in range(Subset): 
            #Choose a random value 
            choice = random.random()
            #Set a sum variable 
            sum = 0
            #For everyone in the population 
            for i in range(pop):
                #Find the member in the population that is to be selected from the following equation 
                sum += 2/pop * (pop - (i+1))/(pop - 1)
                #if this condition is met then we select this member 
                if sum > choice:
                    #add this member to the new population 
                    newPopulation.append(self.population[i])
                    break
        #Set the new population to population list 
        self.population = newPopulation

        ##################################
    # pick a subset of POP based on fitness OR some sort of random or ranked selection
     # USED IN VIDEO TO PRINT EACH STEP DO NOT USE IN PROD 
    #####################################
    def Pselection(self):
        print("SELECTION")
        self.population = sorted(self.population, key=lambda individual: individual.fitness)
        bestChromie = self.population[0]
        self.globalfit.append(bestChromie.fitness)
        if bestChromie.fitness < self.bestChromie.fitness:
            self.bestChromie = bestChromie
        pop = self.pop_size

        #  RANKED ROULETTE SELECTION
        newPopulation = list()
        Subset = int(pop / 2 )
        Subset = Subset + 1 
        for j in range(Subset): 
            choice = random.random()
            print("RANDOM CHOICE")
            print(choice)
            sum = 0
            for i in range(pop):
                sum += 2/pop * (pop - (i+1))/(pop - 1)
                print("Chromosome picked")
                print(sum)
                if sum > choice:
                    print("Parent selected")
                    print(self.population[i])
                    newPopulation.append(self.population[i])
                    break

        self.population = newPopulation



    ####################################
    # make new generation based on parent selection by swapping chromosomes 
    ####################################
    def crossover(self): 
        #Update the generation 
        self.generation = self.generation + 1
        #Create a temp list 
        NewPop = list() 
        #Uniform cross over method chosen
        for i in range(len(self.population)-1): 
            #Create 2 new temp lists for the chromosomes 
            NewChromoC1 = list()
            NewChromoC2 = list()  
            #Pick the 2 parents that will cross over 
            Parent1 = self.population[i]
            Parent2 = self.population[i+1]
            #Create 2 new children to be made from the parents 
            Child1 = individual()
            Child2 = individual()
            #SEt the child size to the parents size 
            Child1.InitChromie(Parent1.getsize())
            Child2.InitChromie(Parent2.getsize())
            #For each of the values in the chromosome 
            for i in range(Parent1.getsize()):
                #Randomly generate a value 
                score = random.random()
                #If the selected score is higher than the cross over rate 
                if score > self.crossover_rate: 
                    #Set a variable to the first parents chromosome at the given index position 
                    bit = Parent1.getChromie()
                    bit = bit[i]
                    #Set a second variable to the second parents chromsome position value 
                    bit2 = Parent2.getChromie()
                    bit2 = bit2[i]
                #Other wise 
                else: 
                    #Set a variable to the second parents chromosome at the given index position 
                    bit = Parent2.getChromie()
                    bit = bit[i]
                    #Set a second variable to the first parents chromsome position value 
                    bit2 = Parent1.getChromie()
                    bit2 = bit2[i]
                #Append these values to the respective childrens chromosome arrays
                NewChromoC1.append(bit)
                NewChromoC2.append(bit2)
            #Ensure that the new chromosomes are numpy arrays 
            NewChromoC1 = np.array(NewChromoC1)
            NewChromoC2 = np.array(NewChromoC2)
            #Set the children chromosome values 
            Child1.SetChromie(NewChromoC1)
            Child2.SetChromie(NewChromoC2)
            #Add these children to the new population 
            NewPop.append(Child1)
            NewPop.append(Child2)
        #Update the population to only be the new population 
        self.population = NewPop
        #IF we generate too many children randomly kill them until the population is controlled 
        while(len(self.population) > self.pop_size): 
            #Pick a random int 
            Kill = random.randint(0,len(self.population))
            #Remove the member from the population 
            self.population.remove(self.population[Kill])
        #Now mutate
        self.mutate()


        ####################################
    # make new generation based on parent selection by swapping chromosomes

     # USED IN VIDEO TO PRINT EACH STEP DO NOT USE IN PROD  
    ####################################
    def Pcrossover(self): 
        print("CROSS OVER ")
        self.generation = self.generation + 1
        NewPop = list() 
        for i in range(len(self.population)-1): 

            NewChromoC1 = list()
            NewChromoC2 = list()  
            print("PICKING PARENTS")
            Parent1 = self.population[i]
            Parent2 = self.population[i+1]
            print("CREATING 2 NEW CHILDREN ")
            Child1 = individual()
            Child2 = individual()
            
            Child1.InitChromie(Parent1.getsize())
            Child2.InitChromie(Parent2.getsize())
            
            for i in range(Parent1.getsize()):
                score = random.random()
                if score > self.crossover_rate: 
                    bit = Parent1.getChromie()
                    bit = bit[i]
                    bit2 = Parent2.getChromie()
                    bit2 = bit2[i]
                else: 
                    bit = Parent2.getChromie()
                    bit = bit[i]
                    bit2 = Parent1.getChromie()
                    bit2 = bit2[i]
                NewChromoC1.append(bit)
                NewChromoC2.append(bit2)
            print("NEW CHILDRENS CHROMOSOMES")
            print("Child1 ")
            print(NewChromoC1)
            print("Child 2 ")
            print(NewChromoC2)
            NewChromoC1 = np.array(NewChromoC1)
            NewChromoC2 = np.array(NewChromoC2)
            Child1.SetChromie(NewChromoC1)
            Child2.SetChromie(NewChromoC2)
            NewPop.append(Child1)
            NewPop.append(Child2)
        self.population = NewPop
        
        while(len(self.population) > self.pop_size): 
            Kill = random.randint(0,len(self.population))
            self.population.remove(self.population[Kill])
        self.Pmutate()


    ###################################
    # introduce random change to each individual in the generation
    ###############################
    def mutate(self):
        #For everyone in the popliuation s
        for i in self.population:
            #Roll a random integer value 
            perc = random.random()
            #If this is higher than the mutation rate do not mutate 
            if perc > self.mutation_rate: 
                #go to the next 
                continue 
            #Otherwise 
            else: 
                #Randomly mutate a position in the chromsome 
                bit = random.randint(0,len(i.getChromie())-1)
                #Set a variable to the chromosome 
                temp = i.getChromie()
                #Update the new chromosomes position value 
                temp[bit] = random.uniform(-self.mutation_range,self.mutation_range)
                #Update the chromosome 
                i.SetChromie(temp)  
        ###################################
    # introduce random change to each individual in the generation

     # USED IN VIDEO TO PRINT EACH STEP DO NOT USE IN PROD 
    ###############################
    def Pmutate(self):
        print("MUTATION")
        for i in self.population:
            perc = random.random()
            if perc > self.mutation_rate: 
                print("NO MUTATION ")
                continue 
            else: 
                print("MUTATION")
                bit = random.randint(0,len(i.getChromie())-1)
                temp = i.getChromie()
                temp[bit] = random.uniform(-self.mutation_range,self.mutation_range)
                print("NEW CHROMOSOME")
                print(temp )
                i.SetChromie(temp)  


if __name__ == '__main__':
    print("Program Start")                
    headers = ["Data set", "layers", "maxGen", "pop_size", "mutation_rate", "mutation_range", "crossover_rate", "loss1", "loss2"]
    filename = 'GA_tuning_resultsFINAL.csv'

    Per = Performance.Results()
    Per.PipeToFile([], headers, filename)

    data_sets = ["soybean", "glass", "abalone","Cancer","forestfires", "machine"] 

    regression_data_set = {
        "soybean": False,
        "Cancer": False,
        "glass": False,
        "forestfires": True,
        "machine": True,
        "abalone": True
    }
    categorical_attribute_indices = {
        "soybean": [],
        "Cancer": [],
        "glass": [],
        "forestfires": [],
        "machine": [],
        "abalone": []
    }

    tuned_0_hl = {
        "soybean": {
            "omega": .5,
            "c1": .1,
            "c2": 5,
            "hidden_layer": []
        },
        "Cancer": {
            "omega": .5,
            "c1": .5,
            "c2": 5,
            "hidden_layer": []
        },
        "glass": {
            "omega": .2,
            "c1": .9,
            "c2": 5,
            "hidden_layer": []
        },
        "forestfires": {
            "omega": .2,
            "c1": 5,
            "c2": .5,
            "hidden_layer": []
        },
        "machine": {
            "omega": .5,
            "c1": .9,
            "c2": 5,
            "hidden_layer": []
        },
        "abalone": {
            "omega": .2,
            "c1": 5,
            "c2": .9,
            "hidden_layer": []
        }
    }

    tuned_1_hl = {
        "soybean": {
            "omega": .5,
            "c1": .5,
            "c2": 1,
            "hidden_layer": [7]
        },
        "Cancer": {
            "omega": .2,
            "c1": .5,
            "c2": 5,
            "hidden_layer": [4]
        },
        "glass": {
            "omega": .2,
            "c1": .9,
            "c2": 5,
            "hidden_layer": [8]
        },
        "forestfires": {
            "omega": .2,
            "c1": 5,
            "c2": 5,
            "hidden_layer": [8]
        },
        "machine": {
            "omega": .5,
            "c1": 5,
            "c2": .5,
            "hidden_layer": [4]
        },
        "abalone": {
            "omega": .2,
            "c1": .1,
            "c2": 5,
            "hidden_layer": [8]
        }
    }

    tuned_2_hl = {
        "soybean": {
            "omega": .5,
            "c1": .9,
            "c2": .1,
            "hidden_layer": [7,12]
        },
        "Cancer": {
            "omega": .2,
            "c1": .5,
            "c2": 5,
            "hidden_layer": [4,4]
        },
        "glass": {
            "omega": .2,
            "c1": .9,
            "c2": 5,
            "hidden_layer": [8,6]
        },
        "forestfires": {
            "omega": .2,
            "c1": .9,
            "c2": 5,
            "hidden_layer": [8,8]
        },
        "machine": {
            "omega": .2,
            "c1": .9,
            "c2": .1,
            "hidden_layer": [7,2]
        },
        "abalone": {
            "omega": .2,
            "c1": 5,
            "c2": 5,
            "hidden_layer": [6,8]
        }
    }
    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    total_counter = 1
    for data_set in data_sets:

        data_set_counter = 1
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

        for j in range(3):
            test_data, test_labels = copy.deepcopy(tenfold_data_and_labels[j])
            #Append all data folds to the training data set
            remaining_data = [x[0] for i, x in enumerate(tenfold_data_and_labels) if i!=j]
            remaining_labels = [y[1] for i, y in enumerate(tenfold_data_and_labels) if i!=j]
            #Store off a set of the remaining dataset 
            X = np.concatenate(remaining_data, axis=1) 
            #Store the remaining data set labels 
            labels = np.concatenate(remaining_labels, axis=1)
            print(data_set, "training data prepared")
            regression = regression_data_set[data_set]
            #If the data set is a regression dataset
            if regression == True:
                #The number of output nodes is 1 
                output_size = 1
            #else it is a classification data set 
            else:
                #Count the number of classes in the label data set 
                output_size = du.CountClasses(labels)
                #Get the test data labels in one hot encoding 
                test_labels = du.ConvertLabels(test_labels, output_size)
                #Get the Labels into a One hot encoding 
                labels = du.ConvertLabels(labels, output_size)

            input_size = X.shape[0]

            data_set_size = X.shape[1] + test_data.shape[1]

            tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]
            for z in range(3):
                hidden_layers = tuned_parameters[z]["hidden_layer"]
                maxgen = [500]
                pops = [500] 
                mr = [.2,.5,.8]
                mra = [10]
                crosss = [.2,.5]
                
                total_trials = 324
                for a in maxgen:
                    for b in pops:
                        for c in mr:
                            for d in mra:
                                for e in crosss:
                                        hyperparameters = {
                                              "maxGen":a,
                                              "pop_size":b,
                                              "mutation_rate": c,
                                              "mutation_range": d,
                                              "crossover_rate": e                                          
                                            }
                                        layers = [input_size] + hidden_layers + [output_size]

                                        nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
                                        nn.set_input_data(X,labels)
                                        total_weights = 0 
                                        for i in range(len(layers)-1):
                                            total_weights += layers[i] * layers[i+1]

                                        ga = GA(hyperparameters, total_weights, nn)

                                        # plt.ion
                                        for gen in range(ga.maxGen): 
                                            ga.fitness()
                                            ga.selection()
                                            ga.crossover()

                                            #plt.plot(list(range(len(ga.globalfit))), ga.globalfit)
                                            ##plt.draw()
                                            ##plt.pause(0.00001)
                                            #plt.clf()

                                        # grab the best solution and set the NN weights
                                        bestSolution = ga.bestChromie.getChromie()
                                        bestWeights = ga.nn.weight_transform(bestSolution)
                                        ga.nn.weights = bestWeights

                                        ################################# new code for ga end ###################################
                                        # plt.ioff()
                                        # plt.plot(list(range(len(ga.globalfit))), ga.globalfit)
                                        # plt.show()
                                        # img_name = data_set + '_l' + str(len(hidden_layers)) + '_pr' + str(a) + '_vr' + str(b) + '_w' + str(c) + '_c' + str(d) + '_cc' + str(e) + '_v' + str(f) + '_ps' + str(g) + '.png'
                                        # plt.savefig('tuning_plots/' + img_name)
                                        # plt.clf()
                                        Estimation_Values = ga.nn.classify(test_data,test_labels)
                                        if regression == False: 
                                            #Decode the One Hot encoding Value 
                                            Estimation_Values = ga.nn.PickLargest(Estimation_Values)
                                            test_labels_list = ga.nn.PickLargest(test_labels)
                                            # print("ESTiMATION VALUES BY GIVEN INDEX (CLASS GUESS) ")
                                            # print(Estimation_Values)
                                        else: 
                                            Estimation_Values = Estimation_Values.tolist()
                                            test_labels_list = test_labels.tolist()[0]
                                            Estimation_Values = Estimation_Values[0]
                                        
                                        Estimat = Estimation_Values
                                        groun = test_labels_list
                                        

                                        Nice = Per.ConvertResultsDataStructure(groun, Estimat)
                                        # print("THE GROUND VERSUS ESTIMATION:")
                                        # print(Nice)
                                    

                                        # headers = ["Data set", "layers", "maxGen", "pop_size", "mutation_rate", "mutation_range", "crossover_rate", "loss1", "loss2"]
                                        Meta = [
                                            data_set, 
                                            len(hidden_layers), 
                                            hyperparameters["maxGen"], 
                                            hyperparameters["pop_size"], 
                                            hyperparameters["mutation_rate"],
                                            hyperparameters["mutation_range"],
                                            hyperparameters["crossover_rate"]
                                            ]
                                        Per.StartLossFunction(regression, Nice, Meta, filename)

                                        print(f"{data_set_counter}/{int(total_trials)/6} {data_set}. {total_counter}/{total_trials} total")
                                        data_set_counter += 1
                                        total_counter += 1







    print("Program End ")
