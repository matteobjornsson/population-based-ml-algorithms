#Written by Matteo Bjornsson and Nick Stone 
#################################################################### MODULE COMMENTS ##############################################################################
# This file is the neural network class, THis file has all of the functionality of a neural network that will handle either classification or regression data sets#
# This program takes in a series of hyper parameters that should be tuned for each different neural network, and assumes that all data being inputted has been nor#
#malized, additionally this program uses sigmoid as the hidden layer activation function, and soft max and cross entropy for classifcation and sigmoid and MSE for#
#for regression This program will calculate both forward pass and back propagation for the nerual network                                                         #
#################################################################### MODULE COMMENTS ##############################################################################

from types import new_class
import numpy as np
import math
import TestData
import DataUtility
import pandas as pd
import matplotlib.pyplot as plt
import time 

class NeuralNetwork:

    #On creation of a Neural Network object do the following 
    def __init__(self, input_size: int, hidden_layers: list, regression: bool, 
                    output_size: int, learning_rate: float, momentum: float ) -> None:
        """
        :param input_size: int. dimension of the data set (number of features in x).
        :param hidden_layers: list. [n1, n2, n3..]. List of number of nodes in 
                                each hidden layer. empty list == no hidden layers.
        :param regression: bool. Is this network estimating a regression output?
        :param output_size: int. Number of output nodes (1 for regression, otherwise 1 for each class)
        :param learning_rate: float. Determines the rate the weights are updated. Should be small.
        :param momentum: float. Determines the fraction of the weight/bias update that is used from last pass
        """ 
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.regression = regression
        self.output_size = output_size
        self.layer_node_count = [input_size] + hidden_layers + [output_size]
        self.layers = len(self.layer_node_count)
        # learning rate
        self.learning_rate = learning_rate
        # weights, biases, and layer outputs are lists with a length corresponding to
        # the number of hidden layers + 1. Therefore weights for layer 0 are found in 
        # weights[0], weights for the output layer are weights[-1], etc. 
        self.weights = self.generate_weight_matrices()
        self.biases = self.generate_bias_matrices()
        # activation_outputs[0] is the input values X, where activation_outputs[1] is the
        # activation values output from layer 1. activation_outputs[-1] represents
        # the final output of the neural network
        self.activation_outputs = [None] * self.layers
        self.layer_derivatives = [None] * self.layers
        self.data_labels = None
        # store old versions of the gradients for momentum
        self.old_bias_derivatives = [None] * self.layers
        self.old_weight_derivatives = [None] * self.layers
        self.momentum = momentum
        #following is used to plot error 
        self.error_y = []
        self.error_x = []
        self.pass_count = 0
        

    ################# INITIALIZATION HELPERS ###################################

    #Function generates weigths sets the object variable intial weigths to the newly generated weight values 
    def generate_weight_matrices(self):
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
        self.initial_weights = weights
        return weights

    #Generate the bias for the given neural network 
    def generate_bias_matrices(self):
        # initialize biases as 0
        # generate the matrices that hold the bias value for each layer. Maybe return a list of matrices?
        # will need 1 bias matrix for 0 hidden layers, 2 for 1 hidden layer, 3 for 2 hidden layer. 
        biases = []
        counts = self.layer_node_count
        for i in range(self.layers):
            if i == 0:
                biases.append([])
            else:
                # initialze a (nodes, 1) dimension matrix for each layer. 
                # layer designated by order of append (position in biases list)
                layer_nodes = counts[i]
                biases.append(0)
        return biases

    #Set the object labels and input data to the data that we are taking in , the data set and the labels 
    def set_input_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        ''' Public method used to set the data input to the network and save the
        ground truth labels for error evaluation. 
        Return: None
        '''
        self.activation_outputs[0] = X
        self.data_labels = labels

    ############################################################
    ################# ACTIVATION FUNCTIONS #####################
    ############################################################

    #function to calculate the sigmoid value 
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        ''' Returns sigmoid function of z: s(z) = (1 + e^(-z))^-1
        :param z: weighted sum of layer, to be passed through sigmoid fn
        Return: matrix 
        '''
        # trim the matrix to prevent overflow
        z[z < -700] = -700
        # return the sigmoid
        return 1 / (1 + np.exp(-z))

    #Function to calculate the derivative of the sigmoid function 
    def d_sigmoid(self, z):
        """ Derivative of the sigmoid function: d/dz s(z) = s(z)(1 - s(z))
        Input: real number or numpy matrix
        Return: real number or numpy matrix.
        """
        return self.sigmoid(z) * (1-self.sigmoid(z))
    
    #Function to calculate the soft max value
    # source: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    def SoftMax(self,Values):
        # trim matrix to prevent overflow
        Values[Values > 700] = 700
        # return softmax calculation
        return np.exp(Values) / np.sum(np.exp(Values), axis=0)


    ################# Error functions #####################
    #Generates the mean squared error for a given ground turth and estimate 
    def mean_squared_error(self, ground_truth: np.ndarray, estimate:np.ndarray) -> float:
        """ takes in matrices, calculates the mean squared error w.r.t. target.
        Input matrices must be the same size. 

        :param ground_truth: matrix holding ground truth for each training example
        :param estimate: matrix holding network estimate for each training example
        """
        m = ground_truth.shape[1]
        return (1/m)* np.sum(np.square(ground_truth - estimate))

    #Function to calculate the cross entropy value 
    def CrossEntropy(self,Ground_Truth,Estimate): 
        #Calculate the number of rows in the data set 
        Num_Samples = Estimate.shape[1]
        # output = self.SoftMax(Ground_Truth)
        #Take the log of the estimate (Make sure its not 0 by adding a small value) and then multiply by ground truth 
        Logrithmic = Ground_Truth * np.log(Estimate + .000000000000001)
        #Return the sum of the logs divided by the number of samples 
        return  - np.sum(Logrithmic) / Num_Samples

    ##################################################################
    ################ FORWARD PASS  ###################################
    ##################################################################

    #Function that generates the net input 
    def calculate_net_input(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> None:
        """ Return Z = W*X + b
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        Return: None
        """
        Z = np.dot(W, X) + b
        return Z
    
    #Function: responsible for calculating the sigmoid activation based on the net input value 
    def calculate_sigmoid_activation(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> None:
        """ Return A = sigmoid(W*X + b)
        :param W: matrix of weights of input values incident to the layer
        :param X: matrix input values incident to the layer
        :param b: matrix of bias for the layer
        Return: None
        """
        Z = self.calculate_net_input(W, X, b)
        A = self.sigmoid(Z)
        return A

    #Function will claculate the forward pass and will update acivation function outputs 
    def forward_pass(self) -> None:
        """ Starting from the input layer propogate the inputs through to the output
        layer. Return a matrix of outputs.
        Return: None
        """
        # iterate through each layer, starting at inputs
        for i in range(self.layers):
            # the activation output is known for the first layer (input data)
            if i == 0:
                continue

            # weights into layer i
            W = self.weights[i]
            # outputs of previous layer into layer i
            A = self.activation_outputs[i-1]
            # bias of layer i
            b = self.biases[i]
            # Calculate the activation output for the layer, store for later access
            #if this is a classification network and i is the output layer, caclulate softmax
            if self.regression == False and i == self.layers -1:
                self.activation_outputs[i] = (
                    #Calculate the softmax function 
                    self.SoftMax(self.calculate_net_input(W, A, b))
                )
            # otherwise activation is always sigmoid
            else: 
                self.activation_outputs[i] = (
                    self.calculate_sigmoid_activation(W, A, b)
                )
        # output of the network is the activtion output of the last layer
        final_estimate = self.activation_outputs[-1]
        #calculate the error w.r.t. the ground truth
        if self.regression == False: 
            error = self.CrossEntropy(self.data_labels,final_estimate)
        else: 
            error = self.mean_squared_error(self.data_labels, final_estimate)
            
        self.pass_count += 1
        # save the error to be plotted over time
        if self.pass_count > 1:
            self.error_y.append(error)
            self.error_x.append(self.pass_count)
        
    ####################################################################
    ############### BACKPROPAGATION  ###################################
    ####################################################################

    #Function will calculate the inner layer derivative 
    def calculate_inner_layer_derivative(self, j: int) -> None:
        """ Calculates the partial derivative of the error with respect to the current
        layer: delta_j = a_j * (1 - a_j) * W^T_j+1 dot delta_j+1
        where j = layer, a_j = activation output matrix of layer j, W_j+1 = weights
        matrix of layer j+1 and delta_j+1 is the derivative matrix of layer j+1.
        Here * means elementwise multiplication, 'dot' means dot product. 

        note: a(1-a) is from the derivative of the sigmoid activation function. 
        This would need to be changed if using a different function. 

        :param j: inner layer for which we are calculating the derivative
        """
        # check that this is not the output layer (derivative is different there)
        assert j != self.layers-1
        # activation outputs of this layer
        a = self.activation_outputs[j]
        # weights of the next layer
        W = self.weights[j+1]
        # partial derivative of the next layer
        delta_jPlusOne = self.layer_derivatives[j+1]
        # calculate the derivative of this layer and return it
        d_layer = (self.d_sigmoid(a) * np.dot(W.T, delta_jPlusOne))
        return d_layer

    #This functuion will calcualte the output layer deritvate 
    def calculate_output_layer_derivative(self) -> None:
        """ Return delta_output = (a - Y) * a * (1 - a) 
        ***** this assumes squared error fn and sigmoid activation ************
        Here a = activation output matrix of output layer, Y = ground truth for input
        examples X. 
        * means elementwise multiplication, 'dot' means dot product. 
        """
        # activation outputs of the output layer
        a = self.activation_outputs[-1]
        # ground truth for comparing the estimates to
        Y = self.data_labels
        
        # calculate the derivative dError/dactivation * dactivation/dnet
        # here (a - Y) is the error fn derivative, a * (1-a) is the sigmoid derivative
        #TODO: May be the cause if numbers are skewedd 
        if self.regression == False: 
            d_layer = (a - Y)
        else: 
            d_layer = (a - Y) * a * (1 - a)
        return d_layer

    #This function will use packprogation for a neural network to update the weights 
    def backpropagation_pass(self):
        """ Starting from the input layer propogate the inputs through to the output
        layer.
        """
        # iterate through each layer, starting from the output layer and calculate
        # the partial derivative of each layer
        for i in reversed(range(1, self.layers)):
            # the last layer is differentiated differently, so it is picked out.
            if i == self.layers - 1:
                self.layer_derivatives[i] = self.calculate_output_layer_derivative()
            else:
                self.layer_derivatives[i] = self.calculate_inner_layer_derivative(i)
            # the goal of backpropagation is to update the weights of each layer.
            # This can be done after calculating the derivative for each layer.
            self.update_weights(i)
            self.update_bias(i)
    #This function will update all of the bias values in the nerual network 
    def update_bias(self, i: int) -> None:
        """ update the bias. The formula: B^i_new = B^i_old - delta_i * learning_rate
        the partial derivative of the bias is the same as the partial derivtive of
        of the layer.
        :param i: layer for which we are updating the bias
        Return: None
        """
        #grab this layer's derivative
        # print(self.layer_derivatives[i].shape)
        m = self.layer_derivatives[i].shape[1]
        delta_i = (1/m) * np.sum(self.layer_derivatives[i], axis=1, keepdims=True)
        # delta_i = self.layer_derivatives[i]
        #update the bias
        # if the first pass has already happened, apply momentum term
        if self.pass_count > 1:
            delta_i_old = self.old_bias_derivatives[i]
            self.biases[i] -= self.learning_rate * (delta_i + self.momentum * delta_i_old)
        # otherwise update bias
        else:
            self.biases[i] -= delta_i * self.learning_rate

        self.old_bias_derivatives[i] = delta_i
    #Function that will update all of the weights in the neural network generated b the formula listed in the comment below 
    def update_weights(self, i: int) -> None:
        """ update the weights. The formula:  W^i_new = W^i_old - dW^i * learning_rate
        where dW^i = delta_i dot a^(i-1).T 
        the partial derivative of the weights is the partial derivtive of the layer
        times the activation values of the previous layer.
        :param i: layer for which we are updating the bias
        Return: None
        """
        # get the partial derivative of this layer
        delta_i = self.layer_derivatives[i]
        # get the previous layer activation values
        a_iMinusOne = self.activation_outputs[i-1].T
        # calculate the change in weights per learning rate
        dWeights = np.dot(delta_i, a_iMinusOne)
        # if this is after the first pass, apply the momentum term
        if self.pass_count > 1:
            dWeights_old = self.old_weight_derivatives[i]
            # adjust the weights as the old values minus the derivative times the learning rate
            self.weights[i] -= self.learning_rate * (dWeights + self.momentum * dWeights_old)
        # otherwise just calculate the weights update without mementum
        else:
            # adjust the weights as the old values minus the derivative times the learning rate
            self.weights[i] -= self.learning_rate * dWeights

        self.old_weight_derivatives[i] = dWeights

    ##################### CLASSIFICATION #######################################
    #Given a numpy array of data and labels return a classification guess for the data set 
    def classify(self, X: np.ndarray, Labels: np.ndarray) -> list:
        """ Starting from the input layer propogate the inputs through to the output
        layer. 
        :param X: test data to be classified
        Return: a list of [ground truth, estimate] pairs.
        """
        #Set the input data from the parameters 
        self.set_input_data(X,Labels)
        #Run the forward pass 
        self.forward_pass()
        #Return the labels from the activation outputs 
        return self.activation_outputs[-1]
    
    #Given an array of probabilities pick the index with the highest set 
    def PickLargest(self, Probabilities):
        # print("Pick largest input:", type(Probabilities), Probabilities.shape, '\n', Probabilities)
        Estimation = list()
        #For every column in the OneHot Matrix
        for i in range(Probabilities.shape[1]):
            #Create an index variable to 0 
            Index = 0 
            #Set the value based on the first probability position 
            Value = Probabilities[0][i] 
            #For each of the rows in the One Hot Matrix
            for j in range(len(Probabilities)):
                #If the probability value is greater than the value above 
                if Probabilities[j][i] > Value: 
                    #Set the new value 
                    Value = Probabilities[j][i]
                    #Create a new index poisition 
                    Index = j 
            #Append the index of the value to the array 
            Estimation.append(Index)
        #Return the array 
        return Estimation
 


if __name__ == '__main__':
    # TD = TestData.TestData()
    # X , labels = TD.classification()
    # this code is for testing many points at once from real data
    df = pd.read_csv(f"./NormalizedData/Cancer.csv")
    D = df.to_numpy()
    labels = D[:, -1]
    labels = labels.reshape(1, labels.shape[0]).T
    D = np.delete(D, -1, 1)
    D = D.T
    X = D
    labels = labels.T
    #labels = labels.T
    input_size = X.shape[0]
    hidden_layers = [input_size]
    learning_rate = 3
    momentum = 0 
    regression = False
    output_size = 3
    NN = NeuralNetwork(
        input_size, hidden_layers, regression, output_size,learning_rate,momentum
    )
    NN.set_input_data(X, labels)
    # print(vars(NN))
    plt.ion()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    for i in range(50000):
        NN.forward_pass()
        NN.backpropagation_pass()
        if i % 1000 == 0:
            plt.plot(NN.error_x, NN.error_y)
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
    plt.ioff()
    plt.plot(NN.error_x, NN.error_y)
    plt.show()
