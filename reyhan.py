import numpy as np
import math
import random
import copy
from decimal import *

class JNN:
    def __init__(self, structure=None, training_inputs=None, training_outputs=None):
        self.structure = structure
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

        self.W = [] #planar weights found in generic MLPs
        self.J = [] #Jump weights
        self.B = [] #Biases

        #now generate the weights, using some initialisation technique
        for l in range(0, len(structure)-1):
            self.W.append(np.random.normal(0, 0.5, size=(structure[l], structure[l+1])))
            self.B.append(np.zeros((self.structure[l+1])))

        for l in range(2, len(structure)):
            self.J.append([])
            for k in range(0, l-1):
                self.J[l-2].append(np.random.normal(0, 0.5, size=(structure[k], structure[l])))

        #self.W = np.array(self.W)

    def run(self, this_input, W, J, B):
        #initialise nodes
        Z = []
        A = []
        Z.append(this_input)
        A.append(this_input)
        #now the initialisation is done, we can carry out the propagation
        #the first layer has no jumper, so just do the matrix multiplication
        Z.append(np.add(np.matmul(this_input, self.W[0]), self.B[0]))
        A.append(self.activate(Z[1]))

        for l in range(2, len(self.structure)-1):
            #the next layers are sums of biases, V.W and V.J
            next_Z = np.add(np.matmul(Z[l-1], self.W[l-1]), self.B[l-1])
            #but the V.J contributions are summed from all previous applicable layers
            for j in range(0, l-1):
                next_Z = np.add(next_Z, np.matmul(Z[j], self.J[l-2][j]))

            Z.append(next_Z)

            #activate this node
            A.append(self.activate(next_Z))

        #the very last layer
        last_Z = np.add(np.matmul(Z[-1], self.W[-1]), self.B[-1])
        #jumpers
        for j in range(0, l-1):
            last_Z = np.add(last_Z, np.matmul(Z[j], self.J[-1][j]))

        return(last_Z)


    def activate(self, vector):
        S = vector ** 2
        return(S)

    def activate_differential(self, vector):
        S = 2 * np.dot(vector, vector)
        return(S)

    def loss(self, sample_inputs, sample_outputs, W, J, B):
        total = 0
        batch_size = len(sample_inputs)
        for i in range(0, batch_size):
            diff = sample_outputs[i] - self.run(sample_inputs[i], W, J, B)
            total += np.dot(diff, diff)

        total = total / batch_size

        return(total)

    def backpropagation(self, sample_inputs, sample_outputs):
        #firstly let's run the network once to fill the nodes with values
        batch_size = len(sample_inputs)
        for i in range(0, batch_size):
            this_input = sample_inputs[i]
            Z = []
            A = []
            Z.append(this_input)
            A.append(this_input)
            #now the initialisation is done, we can carry out the propagation
            #the first layer has no jumper, so just do the matrix multiplication
            Z.append(np.add(np.matmul(this_input, self.W[0]), self.B[0]))
            A.append(self.activate(Z[1]))

            for l in range(2, len(self.structure)-1):
                #the next layers are sums of biases, V.W and V.J
                next_Z = np.add(np.matmul(Z[l-1], self.W[l-1]), self.B[l-1])
                #but the V.J contributions are summed from all previous applicable layers
                for j in range(0, l-1):
                    next_Z = np.add(next_Z, np.matmul(Z[j], self.J[l-2][j]))

                Z.append(next_Z)

                #activate this node
                A.append(self.activate(next_Z))

            #the very last layer
            last_Z = np.add(np.matmul(Z[-1], self.W[-1]), self.B[-1])
            #jumpers
            for j in range(0, l-1):
                last_Z = np.add(last_Z, np.matmul(Z[j], self.J[-1][j]))

            #=====================================================================================

            #we have a fully filled network. Now we can backpropagate

            #============================================================================

            #from the last hidden to the output node


            final_gradient = np.subtract(sample_outputs[i], last_Z)
            final_gradient = final_gradient * (-2 / batch_size)

            self.B[-1] -= final_gradient * self.learning_rate

            dW = np.tensordot(A[-1].T, final_gradient, axes=0)
            self.W[-1] -= dW * self.learning_rate

            for i in range(0, len(self.J[-1])):
                dJ = np.tensordot(A[i].T, final_gradient, axes=0)
                self.J[-1][i] -= dJ * self.learning_rate

            #------------------------------------------------------------------

            #now the second last layer:
            dB = np.matmul(final_gradient, self.W[-1].T)
            dB = dB * self.activate_differential(Z[-1])
            self.B[-2] -= dB * self.learning_rate

            dW = np.tensordot(A[-2].T, dB, axes=0)
            self.W[-2] -= dW * self.learning_rate

            for i in range(0, len(self.J[-2])):
                dJ = np.tensordot(A[i].T, dB, axes=0)
                self.J[-2][i] -= dJ * self.learning_rate

            #------------------------------------------------------------------

            #now repeat until we reach l=2
            for l in range(3, len(self.structure)-1):

                # print(self.structure[-l])
                # print(l)

                dB = np.matmul(dB, self.W[-l+1].T) * self.activate_differential(Z[-l+1])
                self.B[-l] -= dB * self.learning_rate

                dW = np.tensordot(A[-l].T, dB, axes=0)
                # print(self.W[-3])
                # print(dW)
                self.W[-l] -= dW * self.learning_rate

                for i in range(0, len(self.J[-l])):
                    dJ = np.tensordot(A[i].T, dB, axes=0)
                    self.J[-l][i] -= dJ * self.learning_rate

            #what we have learnt: backpropagation can be done separately on each route. The values of nodes are not changed.
            #therefore we can backpropagate each route separately
            #although it would be helpful to borrow the dB for dJ


            #second layer ------------------------------------------------------------------

            dB = np.matmul(dB, self.W[2].T) * self.activate_differential(Z[2])
            self.B[1] -= dB * self.learning_rate

            dW = np.tensordot(A[1].T, dB, axes=0)
            # print(self.W[-3])
            # print(dW)
            self.W[1] -= dW * self.learning_rate

            #first layer ------ directly from the input------------------------------------------------------------

            dB = np.matmul(dB, self.W[1].T) * self.activate_differential(Z[1])
            self.B[0] -= dB * self.learning_rate

            dW = np.tensordot(A[0].T, dB, axes=0)
            # print(self.W[-3])
            # print(dW)
            self.W[0] -= dW * self.learning_rate

            # for i in range(0, len(self.J[-2])):
            #     dJ = np.tensordot(A[i].T, dB, axes=0)
            #     self.J[-2][i] -= dJ * self.learning_rate


    def train(self, training_inputs=None, training_outputs=None,
              default_weights=None, default_jumpers=None, default_biases=None):

        if training_inputs != None:
            self.training_inputs = training_inputs

        if training_outputs != None:
            self.training_outputs = self.training_outputs

        if default_weights != None:
            self.W = default_weights

        if default_biases != None:
            self.B = default_biases

        if default_jumpers != None:
            self.J = default_jumpers

        self.learning_rate = 0.0012

        #benchmark:
        self.benchmark = self.loss(self.training_inputs, self.training_outputs,
                                   self.W, self.J, self.B)
        #print(self.benchmark)

        self.backpropagation(self.training_inputs, self.training_outputs)

        new_loss = self.loss(self.training_inputs, self.training_outputs,
                                   self.W, self.J, self.B)

        #print(new_loss)

        epoch = 0

        while epoch <= 1000:
            self.benchmark = new_loss
            self.backpropagation(self.training_inputs, self.training_outputs)

            new_loss = self.loss(self.training_inputs, self.training_outputs,
                                    self.W, self.J, self.B)

            epoch += 1
            #self.learning_rate = 1 / (1 + 0.01* epoch)
            # if new_loss > self.benchmark:
            #     self.learning_rate = self.learning_rate * 0.9
            # else:
            #     self.learning_rate = self.learning_rate * 1.01
            print(new_loss)


        return(self.W, self.J, self.B)




