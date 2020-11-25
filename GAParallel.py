# this is the parallelized version of GA.py. Parallelized by Matteo Bjornsson, original code written by Nick Stone
################################################################################

import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import multiprocessing
import traceback

class individual:
    
    def __init__(self):
        self.fitness = float('inf')
        #How big should each chromosome be? My initial assumption is the number of feature vectors across the board 
        self.chromosome = [] # some numpy == weights
        self.Size = 0 


    def InitChromie(self,Feature_Size):
        #Loop through each index up until the number of features and just set it to 0 
        self.chromosome = [Feature_Size]
        for i in range(len(self.chromosome)): 
            self.chromosome[i] = 0  
        self.chromosome = np.array(self.chromosome)
        self.Size = Feature_Size

    def setfit(self,fit): 
        self.fitness = fit 
    def getfit(self): 
        return self.fitness 
    def SetChromie(self,Chromos): 
        self.chromosome = Chromos
    def SetSize(self,si): 
        self.Size = si 
    def getsize(self): 
        return self.Size
    def getChromie(self): 
        return self.chromosome 

    def ReturnChromie(self):
        return self.chromosome

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
        #On the creation of a genetic algorithm, we should create a series of random weights in a numpy array that can be fed into the neural network. 
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
    ########################################
    def pfitness(self,) -> float:
        print("FITNESS")
        #Fitness Function will be Mean squared Error
        for i in self.population:  
            fitscore = self.nn.fitness(i.getChromie()) 
            print(fitscore)
            i.setfit(fitscore)
            

    ##################################
    # pick a subset of POP based ranked selection
    #####################################
    def selection(self):

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
            sum = 0
            for i in range(pop):
                sum += 2/pop * (pop - (i+1))/(pop - 1)
                if sum > choice:
                    newPopulation.append(self.population[i])
                    break

        self.population = newPopulation

    
    ####################################
    # make new generation based on parent selection by swapping chromosomes 
    ####################################
    def crossover(self): 
        self.generation = self.generation + 1
        NewPop = list() 
        #{01 12 23 34 }
        #TODO: pick crossover mechanism (uniform?)
        for i in range(len(self.population)-1): 

            NewChromoC1 = list()
            NewChromoC2 = list()  

            Parent1 = self.population[i]
            Parent2 = self.population[i+1]
            
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
        self.mutate()


   

    ###################################
    # introduce random change to each individual in the generation
    ###############################
    def mutate(self):
        for i in self.population:
            perc = random.random()
            if perc > self.mutation_rate: 
                continue 
            else: 
                bit = random.randint(0,len(i.getChromie())-1)
                temp = i.getChromie()
                temp[bit] = random.uniform(-self.mutation_range,self.mutation_range)
                i.SetChromie(temp)  
     


def driver(q, ds: str, data_package: list, regression: bool, perf: Performance, hidden_layers: list, hyper_params: dict, count: int, total_counter:int, total: int):
    print("Job ", ds, count, "started")
    try:
        # init all test data values
        test_data, test_labels, training_data, training_labels, output_size, input_size = data_package
        layers = [input_size] + hidden_layers + [output_size]

        # init neural network
        nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
        nn.set_input_data(training_data, training_labels)

        total_weights = 0 
        for i in range(len(layers)-1):
            total_weights += layers[i] * layers[i+1]
    
        #self, hyperparameters:dict , Total_Weight:int ,NN
        ga = GA(hyper_params,total_weights, nn)
        # plt.ion
        for gen in range(ga.maxGen): 
            ga.fitness()
            ga.selection()
            ga.crossover()
        
        # get best overall solution and set the NN weights
        bestSolution = ga.bestChromie.getChromie()
        bestWeights = ga.nn.weight_transform(bestSolution)
        ga.nn.weights = bestWeights

        # pass the test data through the trained NN
        results = classify(test_data, test_labels, regression, ga, perf)
        # headers = ["Data set", "layers", "pop", "Beta", "CR", "generations", "loss1", "loss2"]

        Meta = [
            ds, 
            len(hidden_layers), 
            hyper_params["maxGen"], 
            hyper_params["pop_size"], 
            hyper_params["mutation_rate"],
            hyper_params["mutation_range"],
            hyper_params["crossover_rate"]
            ]
        results_performance = perf.LossFunctionPerformance(regression, results) 
        data_point = Meta + results_performance
        data_point_string = ','.join([str(x) for x in data_point])
        # put the result on the multiprocessing queue
        q.put(data_point_string)
        print(f"{ds} {count}/{int(total/6)}. {total_counter}/{total}")
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print()
        raise e

def generate_data_package(fold: int, tenfolds: list, regression: bool, du: DataUtility):
    test_data, test_labels = copy.deepcopy(tenfolds[fold])
    remaining_data = [x[0] for i, x in enumerate(copy.deepcopy(tenfolds)) if i!=fold]
    remaining_labels = [y[1] for i, y in enumerate(copy.deepcopy(tenfolds)) if i!=fold]
    #Store off a set of the remaining dataset 
    training_data = np.concatenate(remaining_data, axis=1) 
    #Store the remaining data set labels 
    training_labels = np.concatenate(remaining_labels, axis=1)
    
    if regression == True:
        #The number of output nodes is 1 
        output_size = 1
    #else it is a classification data set 
    else:
        #Count the number of classes in the label data set 
        output_size = du.CountClasses(training_labels)
        #Get the test data labels in one hot encoding 
        test_labels = du.ConvertLabels(test_labels, output_size)
        #Get the Labels into a One hot encoding 
        training_labels = du.ConvertLabels(training_labels, output_size)

    input_size = training_data.shape[0]
    return [test_data, test_labels, training_data, training_labels, output_size, input_size]

def classify(test_data: np.ndarray, test_labels: np.ndarray, regression: bool, ga: GA, perf: Performance):
    estimates = ga.nn.classify(test_data, test_labels)
    if regression == False: 
        #Decode the One Hot encoding Value 
        estimates = ga.nn.PickLargest(estimates)
        ground_truth = ga.nn.PickLargest(test_labels)
    else: 
        estimates = estimates.tolist()
        ground_truth = test_labels.tolist()[0]
        estimates = estimates[0]
    results = perf.ConvertResultsDataStructure(ground_truth, estimates)
    return results

# this function takes the results from the queue that all async jobs write to, and
# writes the jobs to disk. This function is meant to be started as it's own process.
# param q is the multiprocess Manager queue object shared by all jobs. 
def data_writer(q, filename):
    while True:
        with open(filename, 'a') as f:
            data_string = q.get()
            if data_string == 'kill':
                f.write('\n')
                break
            f.write(data_string + '\n')

if __name__ == '__main__':

    headers = ["Data set", "layers", "maxGen", "pop_size", "mutation_rate", "mutation_range", "crossover_rate", "loss1", "loss2"]
    filename = 'GA_results.csv'

    Per = Performance.Results()
    Per.PipeToFile([], headers, filename)

    data_sets = ["soybean", "glass","Cancer","forestfires", "machine", "abalone"] 

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
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": []
        },
        "Cancer": {
            "mutation_rate": .8,
            "crossover_rate": .5,
            "hidden_layer": []
        },
        "glass": {
            "mutation_rate": .5,
            "crossover_rate": .2,
            "hidden_layer": []
        },
        "forestfires": {
            "mutation_rate": .2,
            "crossover_rate": .5,
            "hidden_layer": []
        },
        "machine": {
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": []
        },
        "abalone": {
            "mutation_rate": .5,
            "crossover_rate": .5,
            "hidden_layer": []
        }
    }

    tuned_1_hl = {
        "soybean": {
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": [7]
        },
        "Cancer": {
            "mutation_rate": .5,
            "crossover_rate": .2,
            "hidden_layer": [4]
        },
        "glass": {
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": [8]
        },
        "forestfires": {
            "mutation_rate": .5,
            "crossover_rate": .2,
            "hidden_layer": [8]
        },
        "machine": {
            "mutation_rate": .5,
            "crossover_rate": .2,
            "hidden_layer": [4]
        },
        "abalone": {
            "mutation_rate": .8,
            "crossover_rate": .2,
            "hidden_layer": [8]
        }
    }

    tuned_2_hl = {
        "soybean": {
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": [7,12]
        },
        "Cancer": {
            "mutation_rate": .5,
            "crossover_rate": .2,
            "hidden_layer": [4,4]
        },
        "glass": {
            "mutation_rate": .5,
            "crossover_rate": .5,
            "hidden_layer": [8,6]
        },
        "forestfires": {
            "mutation_rate": .2,
            "crossover_rate": .5,
            "hidden_layer": [8,8]
        },
        "machine": {
            "mutation_rate": .5,
            "crossover_rate": .5,
            "hidden_layer": [7,2]
        },
        "abalone": {
            "mutation_rate": .2,
            "crossover_rate": .2,
            "hidden_layer": [6,8]
        }
    }
    ##############################################
    # START MULTIPROCESS JOB POOL
    ##############################################
    manager = multiprocessing.Manager()
    q = manager.Queue()
    writer = multiprocessing.Process(target=data_writer, args=(q,filename))
    writer.start()

    pool = multiprocessing.Pool()
    ##############################################

    du = DataUtility.DataUtility(categorical_attribute_indices, regression_data_set)
    total_counter = 0
    for data_set in data_sets:
        if data_set != "abalone": continue
        regression = regression_data_set[data_set]
        tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]

        data_set_counter = 0
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

        for j in range(10):
            data_package = generate_data_package(fold=j, tenfolds=tenfold_data_and_labels, regression=regression, du=du)

            for z in range(3):
                if z != 2: continue
                hidden_layers = tuned_parameters[z]["hidden_layer"]

                # these are the parameters that were tuned:
                ############################################
                # popss =[100] # paper suggests 10 * total weight
                # bet = [.5,.8,.2] # note suggested from paper: [.5 , 1]
                # cr = [.1, .3, .8] # note suggested from paper: cr from [0,.3], [.8, 1] if not converging
                # maxgen = [500]

                total_trials = 180

                hyperparameters = {
                        "maxGen":500,
                        "pop_size":500,
                        "mutation_rate": tuned_parameters[z]["mutation_rate"],
                        "mutation_range": 10,
                        "crossover_rate": tuned_parameters[z]["crossover_rate"]                                         
                    }


                pool.apply_async(driver, args=(
                    q, # queue
                    data_set, 
                    data_package,
                    regression,
                    Per,
                    hidden_layers,
                    hyperparameters,
                    data_set_counter,
                    total_counter,
                    total_trials
                ))
                data_set_counter += 1
                total_counter += 1

    ##############################
    # CLOSE THE MULTIPROCESS POOL
    ##############################
    pool.close()
    pool.join()
    q.put('kill')
    writer.join()

