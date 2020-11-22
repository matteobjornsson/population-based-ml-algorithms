import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
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

def driver(q, maxIter: int, ds: str, data_package: list, regression: bool, du: DataUtility, perf: Performance, hidden_layers: list, hyper_params: dict, count: int, total_counter:int, total: int):
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
            pso.update_fitness()
            pso.update_position_and_velocity()
        
        # get best overall solution and set the NN weights
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

if __name__ == '__main__':

    headers = ["Data set", "layers", "omega", "c1", "c2", "vmax", "pop_size", "maxIter", "loss1", "loss2"]
    filename = 'PSO_tuning_take2.csv'

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

        regression = regression_data_set[data_set]
        tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]

        data_set_counter = 0
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

        for j in range(10):
            data_package = generate_data_package(fold=j, tenfolds=tenfold_data_and_labels, regression=regression, du=du)

            for z in range(3):
                hidden_layers = tuned_parameters[z]["hidden_layer"]

                for repeat in range(3):
                    omega = [.2, .5, .8]
                    c1 = [.1, .5, .9, 5]
                    c2 = [.1, .5, .9, 5]
                    vmax = [1]
                    pop_size = [1000, 10000]
                    max_iter = [100, 1000]
                    for a in omega:
                        for b in c1:
                            for c in c2:
                                for d in vmax:
                                    for e in pop_size:
                                        for f in max_iter:
                                            hyperparameters = {
                                                "position_range": 10,
                                                "velocity_range": 1,
                                                "omega": a,
                                                "c1": b,
                                                "c2": c,
                                                "vmax": d,
                                                "pop_size": e,
                                                "max_iter": f                                              
                                                }
                    # hyperparameters = {
                    #     "position_range": 10,
                    #     "velocity_range": 1,
                    #     "omega": tuned_parameters[z]["omega"],
                    #     "c1": tuned_parameters[z]["c1"],
                    #     "c2": tuned_parameters[z]["c2"],
                    #     "vmax": 1,
                    #     "pop_size": 1000                                        
                    #     }
                    # if data_set == "soybean": hyperparameters["vmax"] = 7

                                            pool.apply_async(driver, args=(
                                                q, # queue
                                                hyperparameters["max_iter"], # max iter
                                                data_set, 
                                                data_package,
                                                regression,
                                                du,
                                                Per,
                                                hidden_layers,
                                                hyperparameters,
                                                data_set_counter,
                                                total_counter,
                                                103680
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

