import random
import math
import numpy as np
import sevda


import matplotlib.pyplot as plt
import time
import threading

# #firstly prepare data. Do a sin curve between 0 and pi/2
#
# X = np.linspace(0, math.pi/2, 100)
# Y = np.cos(X)
#
# AI = sevda.JNN(training_inputs=X, training_outputs=Y)
# trained_weights, trained_biases = AI.STD(X, Y)
#
# print(trained_weights, trained_biases)
#
# test_Y = []
# for x in X:
#     y = AI.forward(x, trained_weights, trained_biases)
#     test_Y.append(y)
#
# fig = plt.Figure()
# real = plt.plot(X, Y)
# test = plt.plot(X, test_Y)
#
# def update():
#     time.sleep(1)
#     epoch = 0
#     while True:
#         trained_weights, trained_biases = AI.STD(X, Y)
#         test_Y = []
#         for x in X:
#             y = AI.forward(x, trained_weights, trained_biases)
#             test_Y.append(y)
#         test[0].set_ydata(test_Y)
#         plt.pause(0.0000001)
#         epoch += 1
#         print("Epoch: ", epoch, " ,  Loss: ", AI.loss(X, Y, trained_weights, trained_biases))
# update()
#
# plt.show()

import reyhan

shape = [1, 1, 1, 1, 1, 1]
sample_inputs = []
sample_outputs = []

for j in range(0, 10):
    sample_inputs.append([])
    sample_outputs.append([])

    for i in range(0, 100):
        xi = np.random.uniform(0, math.pi/2, size=(1))
        sample_inputs[j].append(xi)
        sample_outputs[j].append(xi ** 2)

AI = reyhan.JNN(shape, training_inputs=sample_inputs[0], training_outputs=sample_outputs[0])



j=0
weights, jumpers, biases = AI.train(training_inputs=sample_inputs[j], training_outputs=sample_outputs[j])

test_Y = []
for x in sample_inputs[0]:
    y = AI.run(x, weights, jumpers, biases)
    test_Y.append(y)

fig = plt.Figure()
real = plt.scatter(sample_inputs[0], sample_outputs[0])
test = plt.scatter(sample_inputs[0], test_Y)

plt.show()

