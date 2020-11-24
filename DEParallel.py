import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import multiprocessing
import traceback

class individual(): 
 
    def __init__(self, Size):
            self.fitness = float('inf')
            self.Chromie = list() 
            lowerbound = -10 
            upperbound = 10 
            for i in range(Size):
                self.Chromie.append(random.uniform(lowerbound,upperbound))
            self.Chromie = np.array(self.Chromie)
    def getchromie(self): 
        return self.Chromie
    def setchromie(self,chrom):
        self.Chromie = chrom 
    def getfit(self): 
        return self.fitness
    def setfit(self,fitt): 
        self.fitness = fitt


class DE:
    #this is the hyperparameter from mutation

    #####################
    # Initialize the population etc
    ####################
    def __init__(self, hyperparameters:dict , Chromie_Size:int , nn:NeuralNetwork):
        #Hyperparameters v
        self.beta = hyperparameters["beta"]
        self.maxgens = hyperparameters["max_gen"]
        self.pop_size = hyperparameters["population_size"]
        self.probability_of_crossover = hyperparameters["crossover_rate"]
        #Hyperparameters ^

        self.population = list() #TODO: what data structure to use here?
        for i in range(self.pop_size): 
            temp = individual(Chromie_Size)
            self.population.append(temp)
        self.generation = 0
        self.nn = nn 
        self.globalbest = list() 
        self.bestChromie = self.population[0]

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
            while(count < 4): 
                org = random.randint(0,len(self.population)-1)
                if org in nums: 
                    continue 
                nums.append(org)
                count = count +1
            organism = self.population[i]
            nums.remove(i)
            organ1 = self.population[nums[0]]
            organ2 = self.population[nums[1]]
            organ3 = self.population[nums[2]]
            temp = copy.deepcopy(organism.getchromie())
            for j in range(len(organism.getchromie())): 
                x1 = organ1.getchromie()[j]
                x2 = organ2.getchromie()[j]
                x3 = organ3.getchromie()[j]

                b = self.beta
                ColumnTV =  x1 + b * (x2 - x3)      
                coin = random.random() 
                if coin < self.probability_of_crossover: 
                    temp[j] = ColumnTV
                else: 
                    #No crossover 
                    continue 
            fitness = self.fitness(temp)

            if fitness < organism.getfit(): 
                organism.setfit(fitness)
                organism.setchromie(temp)
            if fitness < bestfit: 
                bestfit = fitness
            if fitness < self.bestChromie.fitness:
                self.bestChromie = organism
            
            self.population[i] = organism
        self.globalbest.append(bestfit)

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

        de = DE(hyper_params,total_weights, nn)
        # plt.ion
        for gen in range(de.maxgens): 
            de.mutate_and_crossover()
        
        # get best overall solution and set the NN weights
        bestSolution = de.bestChromie.getchromie()
        bestWeights = de.nn.weight_transform(bestSolution)
        de.nn.weights = bestWeights

        # pass the test data through the trained NN
        results = classify(test_data, test_labels, regression, de, perf)
        # headers = ["Data set", "layers", "pop", "Beta", "CR", "generations", "loss1", "loss2"]

        Meta = [
            ds, 
            len(hidden_layers), 
            hyper_params["population_size"], 
            hyper_params["beta"], 
            hyper_params["crossover_rate"],
            hyper_params["max_gen"]
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

def classify(test_data: np.ndarray, test_labels: np.ndarray, regression: bool, de: DE, perf: Performance):
    estimates = de.nn.classify(test_data, test_labels)
    if regression == False: 
        #Decode the One Hot encoding Value 
        estimates = de.nn.PickLargest(estimates)
        ground_truth = de.nn.PickLargest(test_labels)
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

    headers = ["Data set", "layers", "pop", "Beta", "CR", "generations", "loss1", "loss2"]
    filename = 'DE_tuning_take2.csv'

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
        if data_set == "soybean": continue
        if data_set == "glass": continue 
        if data_set == "abalone": continue
        regression = regression_data_set[data_set]
        tuned_parameters = [tuned_0_hl[data_set], tuned_1_hl[data_set], tuned_2_hl[data_set]]

        data_set_counter = 0
        # ten fold data and labels is a list of [data, labels] pairs, where 
        # data and labels are numpy arrays:
        tenfold_data_and_labels = du.Dataset_and_Labels(data_set)

        for j in range(3):
            data_package = generate_data_package(fold=j, tenfolds=tenfold_data_and_labels, regression=regression, du=du)

            for z in range(3):
                hidden_layers = tuned_parameters[z]["hidden_layer"]

                popss =[500] # paper suggests 10 * total weight
                bet = [.5,.8,.2] # note suggested from paper: [.5 , 1]
                cr = [.1, .3, .8] # note suggested from paper: cr from [0,.3], [.8, 1] if not converging
                maxgen = [500]

                total_trials = 486
                """
                
                                hyperparameters = {
                                    "population_size": 10*total_weights,
                                    "beta": .5,
                                    "crossover_rate": .6, 
                                    "max_gen": 100                                              
                                    }
                """
                for a in popss: 
                    for b in bet:
                        for c in cr: 
                            for d in maxgen: 
                                hyperparameters = {
                                    "population_size": a,
                                    "beta": b,
                                    "crossover_rate": c, 
                                    "max_gen": d                                          
                                    }
                                # def driver(
                                #   q, 
                                #   ds: str, 
                                #   data_package: list,     
                                #   regression: bool, 
                                #   perf: Performance, 
                                #   hidden_layers: list, 
                                #   hyper_params: dict, 
                                #   count: int, 
                                #   total_counter:int, 
                                #   total: int):

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

