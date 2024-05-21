import random
import math
import numpy as np
import sevda
import matplotlib.pyplot as plt
import time
import threading

#firstly prepare data. Do a sin curve between 0 and pi/2

X = np.linspace(0, math.pi/2, 100)
Y = np.sin(X)

AI = sevda.JNN(training_inputs=X, training_outputs=Y)
trained_weights, trained_biases = AI.STD(X, Y)

print(trained_weights, trained_biases)

test_Y = []
for x in X:
    y = AI.forward(x, trained_weights, trained_biases)
    test_Y.append(y)

fig = plt.Figure()
real = plt.plot(X, Y)
test = plt.plot(X, test_Y)

def update():
    time.sleep(1)
    epoch = 0
    while True:
        trained_weights, trained_biases = AI.STD(X, Y)
        test_Y = []
        for x in X:
            y = AI.forward(x, trained_weights, trained_biases)
            test_Y.append(y)
        test[0].set_ydata(test_Y)
        plt.pause(0.1/(1+ epoch * 0.03))
        epoch += 1
        print("Epoch: ", epoch)
update()

plt.show()
