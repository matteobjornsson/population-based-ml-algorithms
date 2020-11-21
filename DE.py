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
            temp = organism.getchromie()
            for j in range(len(organism.getchromie())): 
                ColumnTV= organ1.getchromie()[j] + self.beta * ((organ2.getchromie()[j] * organ2.getchromie()[j]) - (organ3.getchromie()[j] * organ3.getchromie()[j]))      
                coin = random.randint(0,99) + 1 
                if coin > self.probability_of_crossover: 
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
            
            self.population[i] = organism
        self.globalbest.append(bestfit)

    ##################################
    # Main function down here? 
    # Remember, functions we can dispatch as jobs with unique parameters = parallelizeable. 
    #################################
    def driver(self,input_parameters): 
        # Until convergence: 
            # for each individual:
                # mutate and crossover to generate replacement 
                # eval fitness of replacement, keep better of the two
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
                
                pso = DE(layers,total_weights, nn)
                plt.ion
                for gen in range(100): 
                    pso.mutate_and_crossover()
                    

                    plt.plot(list(range(len(pso.globalbest))), pso.globalbest)
                    plt.draw()
                    plt.pause(0.00001)
                    plt.clf()
                ################################# new code for PSO end ###################################
                plt.ioff()
                plt.plot(list(range(len(pso.globalbest))), pso.globalbest)
                plt.show()
                # img_name = data_set + '_l' + str(len(hidden_layers)) + '_pr' + str(a) + '_vr' + str(b) + '_w' + str(c) + '_c' + str(d) + '_cc' + str(e) + '_v' + str(f) + '_ps' + str(g) + '.png'
                # plt.savefig('tuning_plots/' + img_name)
                # plt.clf()
                Estimation_Values = pso.nn.classify(test_data,test_labels)
                if regression == False: 
                    #Decode the One Hot encoding Value 
                    Estimation_Values = pso.nn.PickLargest(Estimation_Values)
                    test_labels_list = pso.nn.PickLargest(test_labels)
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
