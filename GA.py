import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import matplotlib.pyplot as plt

class individual:
    
    def __init__(self):
        self.fitness = 10
        #How big should each chromosome be? My initial assumption is the number of feature vectors across the board 
        self.chromosome = [] # some numpy == weights
        self.Size = 0 


    def InitChromie(Feature_Size):
        #Loop through each index up until the number of features and just set it to 0 
        self.chromosome = [Feature_Size]
        for i in range(Feature_Size): 
            self.chromosome[i] = 0  

    def SetChromie(Chromos): 
        self.chromosome = Chromos
    def SetSize(si): 
        self.Size = si 
    def getsize(): 
        return self.size
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
        self.globalfit = list() 
        self.pop_size = 10 
        #init general population 
        #On the creation of a genetic algorithm, we should create a series of random weights in a numpy array that can be fed into the neural network. 
        #Create an individual object and set the chromosome weight randomly for each of the individuals in the population (pop size)
        for i in range(self.pop_size): 
            #Create a new individual object 
            temp = individual()
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
    def fitness(self,) -> float:
        #Fitness Function will be Mean squared Error
        for i in self.population:  
            fitscore = self.nn.fitness(i.getChromie()) 
            self.fit.append(fitscore)

    ##################################
    # pick a subset of POP based on fitness OR some sort of random or ranked selection
    #####################################
    def selection(self):

 ######################################### Change to be probablistic Chance #############################################
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
        self.globalfit.append(newFitness[0])

    ####################################
    # make new generation based on parent selection by swapping chromosomes 
    ####################################
    def crossover(self): 
        self.generation = self.generation + 1
        NewPop = list()
        i = 0 
        j = 1
        #TODO: pick crossover mechanism (uniform?)
        while(range(len(self.population)-1)):
            if j > len(self.population) -1: 
                break 
            NewChromoC1 = list()
            NewChromoC2 = list()  
            Parent1 = self.population[i]
            Parent2 = self.population[j]
            Child1 = self.individual()
            Child2 = self.individual()
            Child1.InitChromie(Parent1.getsize())
            Child2.InitChromie(Parent2.getsize())
            for i in range(Parent1.getsize()):
                score = random.randint(0,99) + 1
                if score > 50: 
                    bit = Parent1.getChromie()
                    bit = bit[i]
                else: 
                    bit = Parent2.getChromie()
                    bit = bit[i]
                NewChromoC1.append(bit)
                score = random.randint(0,99) + 1
                if score > 50: 
                    bit = Parent1.getChromie()
                    bit = bit[i]
                else: 
                    bit = Parent2.getChromie()
                    bit = bit[i]
                NewChromoC2.append(bit)
            Child1.setChromie(NewChromoC1)
            Child2.setChromie(NewChromoC2)
            NewPop.append(Child1)
            NewPop.append(Child2)
            i = i + 2 
            j = j + 2 
        self.population = NewPop
        while(len(self.population) > self.pop_size): 
            Kill = random.randint(0,len(self.population))
            self.population.remove(self.population[Kill])
        self.mutate()

    ###################################
    # introduce random change to each individual in the generation
    ###############################
    def mutate(self):
        for i in self.population: 
            perc = random.randint(0,99) + 1 
            if perc < 85: 
                continue 
            else: 
                Mutation = GenerateWeights()
                i.setChromie(Mutation)  

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
    headers = ["Data set", "layers", "omega", "c1", "c2", "vmax", "pop_size", "loss1", "loss2"]
    filename = 'GA_experimental_results.csv'

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
    total_counter = 0
    for data_set in data_sets:
        data_set_counter = 0
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

        for j in range(10):
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

                hyperparameters = {
                    "position_range": 10,
                    "velocity_range": 1,
                    "omega": tuned_parameters[z]["omega"],
                    "c1": tuned_parameters[z]["c1"],
                    "c2": tuned_parameters[z]["c2"],
                    "vmax": 1,
                    "pop_size": 1000                                                
                    }
                if data_set == "soybean": hyperparameters["vmax"] = 7

                layers = [input_size] + hidden_layers + [output_size]

                nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
                nn.set_input_data(X,labels)
                total_weights = 0 
                for i in range(len(layers)-1):
                    total_weights += layers[i] * layers[i+1]
                pso = GA(layers,total_weights, total_weights, nn)
                for gen in range(10): 
                    pso.fitness()
                    pso.selection()
                    pso.crossover()

                # plt.ion
                #for epoch in range(pso.max_t):
                ##    pso.update_fitness()
                 #   pso.update_position_and_velocity()
                    # plt.plot(list(range(len(pso.fitness_plot))), pso.fitness_plot)
                    # plt.draw()
                    # plt.pause(0.00001)
                    # plt.clf()
                ################################# new code for PSO end ###################################
                # plt.ioff()
                # plt.plot(list(range(len(pso.fitness_plot))), pso.fitness_plot)
                # img_name = data_set + '_l' + str(len(hidden_layers)) + '_pr' + str(a) + '_vr' + str(b) + '_w' + str(c) + '_c' + str(d) + '_cc' + str(e) + '_v' + str(f) + '_ps' + str(g) + '.png'
                # plt.savefig('tuning_plots/' + img_name)
                # plt.clf()

                Estimation_Values = pso.NN.classify(test_data,test_labels)
                if regression == False: 
                    #Decode the One Hot encoding Value 
                    Estimation_Values = pso.NN.PickLargest(Estimation_Values)
                    test_labels_list = pso.NN.PickLargest(test_labels)
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
            

                # headers = ["Data set", "layers", "omega", "c1", "c2", "vmax", "pop_size"]
                Meta = [
                    data_set, 
                    len(hidden_layers), 
                    hyperparameters["omega"], 
                    hyperparameters["c1"], 
                    hyperparameters["c2"],
                    hyperparameters["vmax"],
                    1000 # pop size
                    ]
                Per.StartLossFunction(regression, Nice, Meta, filename)
                print(f"{data_set_counter}/30 {data_set}. {total_counter}/180")
                data_set_counter += 1
                total_counter += 1







    print("Program End ")
