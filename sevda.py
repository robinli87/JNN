import math
import random
#import numpy as np
#import copy

class JNN:
    def __init__(self, structure=[1, 1, 1], training_inputs=None, training_outputs=None):
        self.structure = structure
        self.learning_rate = 0.1
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.dt = 0.00001
        self.alpha = 0


        self.weights = [random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)]
        self.biases = [0, 0]
        self.weight_momentum = [0, 0, 0]
        self.biases_momentum = [0, 0]

    def forward(self, local_input, weights, biases):

        z1 = weights[0] * local_input + biases[0]
        a1 = self.activate(z1)
        out = weights[1] * a1 + weights[2] * local_input + biases[1]

        return(out)

    def activate(self, Z):
        #let's do tanh'
        A = Z ** 2
        return(A)

    def loss(self, local_inputs, local_outputs, weights, biases):
        batch_size = len(local_inputs)
        total = 0
        for i in range(0, batch_size):
            diff = local_outputs[i] - self.forward(local_inputs[i], weights, biases)
            total += diff ** 2 / batch_size
        return(total)

    def STD(self, local_inputs, local_outputs):

        #compute gradients by approximating differentiation from first principle

        for i in range(0, len(self.weights)):
            wcp1 = self.copy(self.weights)
            wcp2 = self.copy(self.weights)
            wcp1[i] += self.dt
            up_loss = self.loss(local_inputs, local_outputs, wcp1, self.biases)
            wcp2[i] -= self.dt
            down_loss = self.loss(local_inputs, local_outputs, wcp2, self.biases)
            gradient = (up_loss - down_loss) / (2 * self.dt)
            #print(up_loss, down_loss, gradient)

            self.weights[i] -= self.learning_rate * (gradient + self.weight_momentum[i] * self.alpha)
            self.weight_momentum[i] = gradient

        for i in range(0, len(self.biases)):
            bcp1 = self.copy(self.biases)
            bcp2 = self.copy(self.biases)
            bcp1[i] += self.dt

            up_loss = self.loss(local_inputs, local_outputs, self.weights, bcp1)
            bcp2[i] -= self.dt
            down_loss = self.loss(local_inputs, local_outputs, self.weights, bcp2)
            gradient = (up_loss - down_loss) / (2 * self.dt)

            self.biases[i] -= self.learning_rate * (gradient + self.biases_momentum[i] * self.alpha)
            self.biases_momentum[i] = gradient

        return(self.weights, self.biases)

    def copy(self, arr):
        new = []
        for i in range(0, len(arr)):
            new.append(arr[i])
        return(new)

    def train(self, weights=None, biases=None):

        #check if the user has supplied some pre-trained weights and biases
        if weights == None:
            self.weights = [random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)]
        if biases == None:
            self.biases = [0, 0]

        #first calculate a benchmark loss caused by our given or randomly initialised weights and biases
        self.benchmark = self.loss(self.training_inputs, self.training_outputs, self.weights, self.biases)

        #perform a first stochastic gradient descent
        self.STD(self.training_inputs, self.training_outputs)

        #calculate new loss
        new_loss = self.loss(self.training_inputs, self.training_outputs, self.weights, self.biases)
        print("Benchmark loss: ", self.benchmark)
        print("New loss: ", new_loss)

        epoch = 0

        #iterate training to improve our hyperparameters
        while True:
            #set the benchmark to this improved loss
            self.benchmark = new_loss

            #stochastic gradient descent again
            self.STD(self.training_inputs, self.training_outputs)

            #evaluate the new loss
            new_loss = self.loss(self.training_inputs, self.training_outputs, self.weights, self.biases)
            print("New loss: ", new_loss)

            epoch += 1

            if epoch > 1000:
                break#quit if we've done more than 1000 cycles because I have no patience :D :P

        return(self.weights, self.biases)



