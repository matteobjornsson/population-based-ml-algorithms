# Author: Matteo Bjornsson
################################################################################
# This is the parallelized driver for running the PSO experiment. This file 
# grabs data sets from "./NormalizedData" and for each data set generates 
# tenfolds, and runs PSO to train each of the 0,1,2 layer neural networks. 
################################################################################

import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import pandas as pd
import copy
import multiprocessing
import traceback

class Particle:
    '''
    This class represents a particle in the swarm. Each particle maintains a 
    current position and velocity as well as a running record of its personal best
    position and fitness. 
    '''

    def __init__(self, position_range: float, velocity_range: float, layers: list):
        '''
        :param position_range: float. indicates the bounds for the random uniform
        distribution used to generate initial positions.
        :param velocity_range: float. same as above but for velocity.
        :param layers: list. Expected format: [input nodes, h1_nodes, h2_nodes, output_nodes]
        h1 and h2 can be missing but there must be an input and output: [input, output]
        '''
        # the total weights are represented by the number of weights needed to be 
        # estimated for the NN, which is the sum of each layer multiplied by the next layer
        total_weights = 0
        for i in range(len(layers)-1):
            total_weights += layers[i] * layers[i+1]
        # initialize each weight from a random uniform distribution
        self.position = np.random.uniform(-position_range, position_range, total_weights)
        # same for velocity
        self.velocity = np.random.uniform(-velocity_range, velocity_range, total_weights)
        # we are minimizing our fitness, so it is initialized as pos infinity
        self.fitness = float('inf')
        # current best is initial best
        self.pbest_fitness = self.fitness
        self.pbest_position = None


class PSO:
    '''
    this is the driver class for the PSO. For the given number of iterations or stopping condition
    it will update particle positions and velocities and track the global best position
    which is the position with the best fitness so far. 
    '''

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, layers: list, hyperparameters: dict, NN:NeuralNetwork, maxIter: int):
        #init general population 
            # random weight values, weight matrix is numpy array, matches network architecture
            # use similar weight init function as from NN

        # INIT BOTH *POSITION* (WEIGHTS) AND *VELOCITY*

        
        ############### HYPERPARAMETERS: ################
        # dictates the range of possible initial positions and velocities
        self.position_range = hyperparameters["position_range"]
        self.velocity_range = hyperparameters["velocity_range"]
        # size of inertia
        self.omega = hyperparameters["omega"]
        # size of cognitive component
        self.c1 = hyperparameters["c1"]
        # size of social component
        self.c2 = hyperparameters["c2"]
        # max possible velocity
        self.vmax = hyperparameters["vmax"]
        # popultion size
        self.pop_size = hyperparameters["pop_size"]
        ###################################################
        # store the list of nodes per layer
        self.layers = layers
        self.NN = NN

        # generate the swarm population
        self.population = [] 
        for i in range(self.pop_size):
            self.population.append(Particle(self.position_range, self.velocity_range, layers))
        # track the global best fitness. Initialized as inf as this is a min probelem
        self.gbest_fitness = float('inf')
        self.gbest_position = None
        # number of iterations 
        self.max_t = maxIter


        # fitness plotting:
        self.fitness_plot = []

    def update_position_and_velocity(self):
        # iterate over each particle
            # update v and x using equations from class
            # x_(t+1) = x_t + v_(t+1)
            # v_(t+1) = w*v_t + c1*r1*(pb_t - x_t) + c2*r2*(gb_t - x_t)
        # use velocity clamping on v_max
        for p in self.population:
            # assign variables to improve readability
            v = p.velocity
            w = self.omega
            c1 = self.c1
            r1 = random.uniform(0,1)
            c2 = self.c2
            r2 = random.uniform(0,1)
            pb = p.pbest_position
            gb = self.gbest_position
            x = p.position

            # calculate the new velocity
            new_v = w*v + c1*r1*(pb - x) + c2*r2*(gb - x)

            # clamp velocity to vmax if greater than vmax or less than -vmax.
            # these two lines use numpy functions to select values for which the 
            # conditional is true and set them to vmax
            new_v[new_v > self.vmax] = self.vmax
            new_v[new_v < -self.vmax] = -self.vmax

            # update the new velocity and position
            p.velocity = new_v
            p.position += new_v

    ########################################
    # Evaluate the fitness of an individual
    ########################################
    def update_fitness(self) -> None:
    # for all particles, this method applies the individual's weights to the NN, 
    # feeds data set through and sets the fitness to the error of forward pass

        for p in self.population:
            # run the dataset through the NN with the particle's weights to get fitness
            fitness = self.NN.fitness(p.position)
            # update personal best
            if p.pbest_fitness > fitness:
                p.pbest_fitness = fitness
                p.pbest_position = p.position
            # update global best
            if self.gbest_fitness > fitness:
                self.gbest_fitness = fitness
                self.gbest_position = p.position
            # update particle fitness
            p.fitness = fitness

        # track global best over time each iteration
        self.fitness_plot.append(self.gbest_fitness)


# this method is the target for spawning PSO jobs asynchronously. 
################################################################################
def driver(q, maxIter: int, ds: str, data_package: list, regression: bool, perf: Performance, hidden_layers: list, hyper_params: dict, count: int, total_counter:int, total: int):
    '''
    q: a multiprocessing manager Queue object to pass finished data to
    maxIter: int. number of epochs to run PSO for
    ds: string. name of the data set
    data_package: list. see the data_package method below. Provides all the data needed to run the experiment
    regression: boolean. Is this a regression task?
    perf: Performance type object, used to process estimates vs ground truth
    hidden_layers: list. tells the algorithm how many layers there are in the NN architecture
    hyper_params: dictionary of hyperparameters. useful for tuning.
    count: int. the job number for this data set
    total_counter: int. the job number wrt the total run
    total: int. the total number of jobs across all data sets. used for progress printing. 
    '''
    
    print("Job ", ds, count, "started")
    # multiprocessor in python supresses exceptions when workers fail. This try catch block forces it to raise an exception and stack trace for debugging. 
    try:
        # init all test data values
        test_data, test_labels, training_data, training_labels, output_size, input_size = data_package
        layers = [input_size] + hidden_layers + [output_size]

        # init neural network
        nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
        nn.set_input_data(training_data, training_labels)

        # initi PSO and train it
        pso = PSO(layers, hyper_params, nn, maxIter)
        for epoch in range(pso.max_t):
            print("job", count, "generation", epoch)
            pso.update_fitness()
            pso.update_position_and_velocity()
        
        # get best overall solution from the PSO and set the NN weights
        bestSolution = pso.gbest_position
        bestWeights = pso.NN.weight_transform(bestSolution)
        pso.NN.weights = bestWeights

        # pass the test data through the trained NN
        results = classify(test_data, test_labels, regression, pso, perf)

        # headers = ["Data set", "layers", "omega", "c1", "c2", "vmax", "pop_size", "maxIter", "loss1", "loss2"]
        Meta = [
            ds, 
            len(hidden_layers), 
            hyper_params["omega"], 
            hyper_params["c1"], 
            hyper_params["c2"],
            hyper_params["vmax"],
            hyper_params["pop_size"],
            hyper_params["max_iter"]
            ]
        # get the performance of the network w.r.t. the ground truth
        results_performance = perf.LossFunctionPerformance(regression, results) 
        # construct the data point to be written to disk via csv file
        data_point = Meta + results_performance
        data_point_string = ','.join([str(x) for x in data_point])
        # put the result on the multiprocessing queue
        q.put(data_point_string)
        # status update
        print(f"{ds} {count}/{int(total/6)}. {total_counter}/{total}")

    # if something goes wrong raise an exception 
    except Exception as e:
        print('Caught exception in worker thread')

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()
        print()
        raise e

# this method condenses all data preparation for the algorithm
# returns a list of objects:
# [test_data, test_labels, training_data, training_labels, output_size, input_size]
###############################################################
def generate_data_package(fold: int, tenfolds: list, regression: bool, du: DataUtility):
    # get the fold we are going to use for testing 
    test_data, test_labels = copy.deepcopy(tenfolds[fold])
    # squish the rest of the data and ground truth labels into one numpy array, respectively
    remaining_data = [x[0] for i, x in enumerate(copy.deepcopy(tenfolds)) if i!=fold]
    remaining_labels = [y[1] for i, y in enumerate(copy.deepcopy(tenfolds)) if i!=fold]
    training_data = np.concatenate(remaining_data, axis=1) 
    training_labels = np.concatenate(remaining_labels, axis=1)
    # determine how many output nodes the network has (1 if regression)
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

# this method takes in a PSO object, trained neural network, and test data and classifies the test data using the NN.
# the results returned are the performance loss functions
def classify(test_data: np.ndarray, test_labels: np.ndarray, regression: bool, pso: PSO, perf: Performance):
    estimates = pso.NN.classify(test_data, test_labels)
    if regression == False: 
        #Decode the One Hot encoding Value 
        estimates = pso.NN.PickLargest(estimates)
        ground_truth = pso.NN.PickLargest(test_labels)
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

# this is the main function that runs the PSO algorithm and experiment
if __name__ == '__main__':

    headers = ["Data set", "layers", "omega", "c1", "c2", "vmax", "pop_size", "maxIter", "loss1", "loss2"]
    filename = 'PSO_results.csv'

    # prepare the performance object (also used to write results to file)
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

    ###############################################
    # TUNED HYPERPARAMETERS
    ###############################################
    tuned_0_hl = {
        "soybean": {
            "omega": .4,
            "c1": 3,
            "c2": 3,
            "hidden_layer": []
        },
        "Cancer": {
            "omega": .1,
            "c1": 3,
            "c2": .9,
            "hidden_layer": []
        },
        "glass": {
            "omega": .1,
            "c1": .5,
            "c2": 3,
            "hidden_layer": []
        },
        "forestfires": {
            "omega": .4,
            "c1": .5,
            "c2": .5,
            "hidden_layer": []
        },
        "machine": {
            "omega": .4,
            "c1": 3,
            "c2": .9,
            "hidden_layer": []
        },
        "abalone": {
            "omega": .1,
            "c1": .9,
            "c2": 3,
            "hidden_layer": []
        }
    }

    tuned_1_hl = {
        "soybean": {
            "omega": .4,
            "c1": 3,
            "c2": 3,
            "hidden_layer": [7]
        },
        "Cancer": {
            "omega": .4,
            "c1": .9,
            "c2": .1,
            "hidden_layer": [4]
        },
        "glass": {
            "omega": .1,
            "c1": 3,
            "c2": 3,
            "hidden_layer": [8]
        },
        "forestfires": {
            "omega": .1,
            "c1": .5,
            "c2": .1,
            "hidden_layer": [8]
        },
        "machine": {
            "omega": .1,
            "c1": 3,
            "c2": 3,
            "hidden_layer": [4]
        },
        "abalone": {
            "omega": .1,
            "c1": .1,
            "c2": 3,
            "hidden_layer": [8]
        }
    }

    tuned_2_hl = {
        "soybean": {
            "omega": .4,
            "c1": .9,
            "c2": 3,
            "hidden_layer": [7,12]
        },
        "Cancer": {
            "omega": .4,
            "c1": 3,
            "c2": .5,
            "hidden_layer": [4,4]
        },
        "glass": {
            "omega": .1,
            "c1": .1,
            "c2": 3,
            "hidden_layer": [8,6]
        },
        "forestfires": {
            "omega": .1,
            "c1": .1,
            "c2": .1,
            "hidden_layer": [8,8]
        },
        "machine": {
            "omega": .1,
            "c1": .5,
            "c2": 3,
            "hidden_layer": [7,2]
        },
        "abalone": {
            "omega": .1,
            "c1": .9,
            "c2": 3,
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
    previous_trials = pd.read_csv(filename)

    total_counter = 0
    # iterate over every data set
    for data_set in data_sets:
        # look up if the data set is a regression task
        regression = regression_data_set[data_set]
        # collect the parameters from above
        tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]

        data_set_counter = 0
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)
        # once for each of the ten folds
        for j in range(10):
            # produce the training and test data
            data_package = generate_data_package(fold=j, tenfolds=tenfold_data_and_labels, regression=regression, du=du)
            # once for each number of hidden layers in the network
            for z in range(3):
                hidden_layers = tuned_parameters[z]["hidden_layer"]
                # 6 data sets * 10 folds * 3 layers
                total_trials = 180

                hyperparameters = {
                    "position_range": 10,
                    "velocity_range": 1,
                    "omega": tuned_parameters[z]["omega"],
                    "c1": tuned_parameters[z]["c1"],
                    "c2": tuned_parameters[z]["c2"],
                    "vmax": 1,
                    "pop_size": 100,
                    "max_iter": 500                                              
                    }
                ################################################################
                # # the following code is used to rescue partially completed runs
                #
                # # check if we have already done this hyperparameter set:
                # skip = False
                # for i in range(len(previous_trials)):
                #     try:
                #         # try to pick out all variables
                #         v_data_set = str(previous_trials['Data set'][i])
                #         v_omega = float(previous_trials['omega'][i])
                #         v_c1 = float(previous_trials['c1'][i])
                #         v_c2 = float(previous_trials['c2'][i])
                #         v_vmax = float(previous_trials['vmax'][i])
                #         v_pop = float(previous_trials['pop_size'][i])
                #         v_max_iter = float(previous_trials['maxIter'][i])
                #         # check if the current hyperparameter set already exists in the csv
                #         if (
                #             data_set == v_data_set and 
                #             v_omega == a and
                #             v_c1 == b and
                #             v_c2 == c and
                #             v_vmax == d and
                #             v_pop == e and
                #             v_max_iter == f):
                #             # if it exists in the csv, then set the skip flag to true
                #             skip = True
                #             break
                #     except:
                #         # in the case that a line is unexpected text, skip just that line in the csv
                #         print("csv check: row not recognized")
                #         continue
                # # if the current set of hyperparameters was found in the csv, skip will be true, so skip this hyperparameter set
                # if skip:
                #     continue
                #############################################################

                # spawn a PSO instance per fold, data set, and layer
                pool.apply_async(driver, args=(
                    q, # queue
                    hyperparameters["max_iter"], # max iter
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

