#Just show the average of the ten runs and the first bullet point 
import random
import Performance
from NeuralNetwork import NeuralNetwork
import DataUtility
import numpy as np
import copy
import matplotlib.pyplot as plt
import DE 
import GA 
import PSO 
import VideoNN
import time 



def main(): 
    print("Program Start")
    headers = ["Data set", "layers", "pop", "Beta", "CR", "generations", "loss1", "loss2"]
    filename = 'DE_experimental_resultsFINAL.csv'

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
        if data_set != 'soybean':
            continue 
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

                layers = [input_size] + hidden_layers + [output_size]

                nn = NeuralNetwork(input_size, hidden_layers, regression, output_size)
                nn.set_input_data(X,labels)

                total_weights = 0 
                for i in range(len(layers)-1):
                    total_weights += layers[i] * layers[i+1]
            
                hyperparameters = {
                    "population_size": 10*total_weights,
                    "beta": .5,
                    "crossover_rate": .6, 
                    "max_gen": 100                                              
                    }
                hyperparameterss = {
                                                "maxGen":100,
                                                "pop_size":100,
                                                "mutation_rate": .5,
                                                "mutation_range": 10,
                                                "crossover_rate": .5
                                                }
                hyperparametersss = {
                    "position_range": 10,
                    "velocity_range": 1,
                    "omega": .1, 
                    # tuned_parameters[z]["omega"],
                    "c1": .9,
                    # tuned_parameters[z]["c1"],
                    "c2": .1,
                    # tuned_parameters[z]["c2"],
                    "vmax": 1,
                    "pop_size": 1000,
                    "max_t": 50                                          
                    }
                de = DE.DE(hyperparameters,total_weights, nn)
                ga = GA.GA(hyperparameterss, total_weights, nn)
                pso = PSO.PSO(layers, hyperparametersss, nn)
                learning_rate = 3
                momentum = 0 
                VNN = VideoNN.NeuralNetworks(input_size, hidden_layers, regression, output_size,learning_rate,momentum)

                 
                for gen in range(de.maxgens): 
                    de.mutate_and_crossover()
                     
                
                for gen in range(ga.maxGen): 
                    ga.fitness()
                    ga.selection()
                    ga.crossover()
                   

                counter = 0 
                for epoch in range(pso.max_t):
                    pso.update_fitness()
                    pso.update_position_and_velocity()

                
                   # plt.plot(list(range(len(de.globalbest))), de.globalbest)
                   # plt.draw()
                   # plt.pause(0.00001)
                    #plt.clf()
                # get the best overall solution and set the NN to those weights
                #DE
                bestSolution = de.bestChromie.getchromie()
                bestWeights = de.nn.weight_transform(bestSolution)
                de.nn.weights = bestWeights
                #GA


                #PS


                #   ################################ new code for de end ###################################
                # plt.ioff()
                # plt.plot(list(range(len(de.globalbest))), de.globalbest)
                # plt.show()
                # img_name = data_set + '_l' + str(len(hidden_layers)) + '_pr' + str(a) + '_vr' + str(b) + '_w' + str(c) + '_c' + str(d) + '_cc' + str(e) + '_v' + str(f) + '_ps' + str(g) + '.png'
                # plt.savefig('tuning_plots/' + img_name)
                # plt.clf()
                Estimation_Values = de.nn.classify(test_data,test_labels)
                if regression == False: 
                    #Decode the One Hot encoding Value 
                    Estimation_Values = de.nn.PickLargest(Estimation_Values)
                    test_labels_list = de.nn.PickLargest(test_labels)
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
            

                # headers = ["Data set", "layers", "pop", "Beta", "CR", "generations", "loss1", "loss2"]
                Meta = [
                    data_set, 
                    len(hidden_layers), 
                    hyperparameters["population_size"], 
                    hyperparameters["beta"], 
                    hyperparameters["crossover_rate"],
                    hyperparameters["max_gen"]
                    ]

                Per.StartLossFunction(regression, Nice, Meta, filename)
                print(f"{data_set_counter}/30 {data_set}. {total_counter}/180")
                data_set_counter += 1
                total_counter += 1
                print("DEMO FINISHED")
                time.sleep(10000)

    print("Program End ")




main() 