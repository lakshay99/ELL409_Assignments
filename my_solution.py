import random
import numpy as np



def sqe_loss(expected_out, actual_out):
    return (np.sum((expected_out - actual_out)**2))/2

def sqe_loss_der():


def sigmoid(z):  # input is a numpy vector
    return (1/(1+np.exp(-z)))

def sigmoid_derivative(z): # input is a numpy vector
    temp = sigmoid(z)
    return np.multiply(temp,(1-temp))


class Neural_network:   #ToDo: Add the choice of activation function to the user
    def __init__(self, num_layers, num_neurons, num_inputs, num_outputs):   #num_layers is the number of hidden layers + 1, num_neurons is the number of neurons in each layer
        self.num_layers = num_layers    #assuming atleast two hidden layer present
        self.num_neurons = num_neurons  #there will be one extra neuron to accomodate for the bias term
        self.hidden_weights = np.rand(num_layers-2, num_neurons+1, num_neurons+1)  # weights[i][j][k] gives weight between jth neuron of (i-1)th and kth neuron of ith layer, -1 corresponds to input
        self.input_weights = np.rand(num_inputs + 1, num_neurons) # input_weights[i][j] gives weight b/w ith neuron of input and jth neurons of first output layer
        self.output_weights = np.rand(num_neurons+1, num_outputs)
        self.hidden_weights_derivatives = np.zeros(num_layers-2, num_neurons+1, num_neurons+1)  # weights[i][j][k] gives weight between jth neuron of (i-1)th and kth neuron of ith layer, -1 corresponds to input
        self.input_weights_derivatives = np.zeros(num_inputs + 1, num_neurons) # input_weights[i][j] gives weight b/w ith neuron of input and jth neurons of first output layer
        self.output_weights_derivatives = np.zeros(num_neurons+1, num_outputs)
        self.hidden_neuron_outputs = np.ones((num_neurons+1, num_layers-1))
        self.output_layer_outputs = np.zeros(num_outputs)
        self.hidden_neuron_output_derivatives =  np.zeros((num_neurons+1, num_layers-1)) #Not much significance of using zeros
        self.output_layer_output_derivatives = np.zeros(num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def feedforward(self, input_vector):
        #To set the outputs for first hidden layer
        for y in range(0, self.num_neurons): # As we dont need to set the output for the neuron used for biasing
            self.hidden_neuron_outputs[y][0] = 0
            for z in range(0, self.num_inputs):
                self.hidden_neuron_outputs[y][0] += self.input_weights[z][y]*input_vector[z]
            self.hidden_neuron_outputs[y][0] += self.input_weights[self.num_inputs][y]
            #Activation remaining
        temp = self.hidden_neuron_outputs[ :-1, 0]
        self.hidden_neuron_outputs[ :-1, 0] = sigmoid(temp)
        self.hidden_neuron_output_derivatives[ :-1, 0] = sigmoid_derivative(temp)

        for x in range(1, self.num_layers-1):
            for y in range(0, self.num_neurons): # As we dont need to set the output for the neuron used for biasing
                self.hidden_neuron_outputs[y][x] = 0
                for z in range(0, self.num_neurons):
                    self.hidden_neuron_outputs[y][x] += self.hidden_weights[x-1][z][y]*self.hidden_neuron_outputs[z][x-1]
                self.hidden_neuron_outputs[y][x] += self.hidden_weights[x-1][self.num_neurons][y]
            temp = self.hidden_neuron_outputs[ :-1, x]
            self.hidden_neuron_outputs[ :-1, x] = sigmoid(temp)
            self.hidden_neuron_output_derivatives[ :-1, x] = sigmoid_derivative(temp)
                #Activation remaining

        for y in range(0, self.num_outputs):
            self.output_layer_outputs[y] = 0
            for z in range(0, self.num_neurons):
                self.output_layer_outputs[y] += self.output_weights[z][y]*self.hidden_neuron_outputs[z][num_layers-2]
            self.output_layer_outputs[y] += self.output_weights[self.num_neurons][y]
            #Activation remaining
        temp = self.output_layer_outputs
        self.output_layer_outputs = sigmoid(temp)
        self.output_layer_output_derivatives = sigmoid_derivative(temp)

    def backprop(self, input_vector, actual_output_vector, learning_rate):  #Sets the derivatives equal to new value
        for x in range(0, self.num_outputs):
            for y in range(0, self.num_neurons+1): #Assuming SQE loss
                self.output_weights_derivatives[y][x] = (self.output_layer_outputs[x] - actual_output_vector[x])*self.output_layer_output_derivatives[x]*self.hidden_neuron_outputs[y][self.num_layers-2]

        for x in range(0, self.num_neurons):
            for y in range(0, self.num_neurons + 1):
                temp = 0
                for z in range(0, num_outputs):
                    temp += self.output_weights_derivatives[x][z]*self.output_weights[x][z]
                self.hidden_weights_derivatives[self.num_layers-3][y][x] = temp*self.hidden_neuron_output_derivatives[x][self.num_layers-1]*self.hidden_neuron_outputs[y][self.num_layers-2]/self.hidden_neuron_outputs[x][self.num_layers-1]

        for i in range(self.num_layers-3, 0, -1):
            for x in range(0, self.num_neurons):
                for y in range(0, self.num_neurons+1):
                    temp = 0
                    for z in range(0, num_neurons):
                        temp += self.hidden_weights[i][x][z]*self.hidden_weights_derivatives[i][x][z]
                    self.hidden_weights_derivatives[i-1][y][x] = temp*self.hidden_neuron_output_derivatives[x][i+1]*self.hidden_neuron_outputs[y][i]/self.hidden_neuron_outputs[x][i+1]

        for x in range(0, self.num_neurons):
            for y in range(0, num_inputs + 1):
                temp = 0
                for z in range(0, num_neurons):
                    temp += self.hidden_weights_derivatives[0][x][z]*self.hidden_weights[0][x][z]
                self.input_weights_derivatives[y][x] = temp*self.hidden_neuron_output_derivatives[x][0]*self.input_vector[y]/self.hidden_neuron_outputs[x][0]

        self.hidden_weights -= learning_rate*self.hidden_weights_derivatives
        self.input_weights -= learning_rate*self.input_weights_derivatives
        self.output_weights -= learning_rate*self.output_weights_derivatives


    def gradient_descent(self, input_vectors, output_vectors, learning_rate, num_inputs): #temporarily doing like this
        for x in range(0, 10):
            for i in range(0, num_inputs):
                feedforward(input_vectors[i])
                backprop(input_vectors[i], output_vectors[i], learning_rate)
