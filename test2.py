import tensorflow as tf
import numpy as np
import math

from net import Network
from net import train
from display import display

#Load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels 

#create subset of training data including only 1's and 5's
new_trX1 = np.matrix([[]])
new_trX5 = np.matrix([[]])
new_trY1 = np.matrix([[]])
new_trY5 = np.matrix([[]])

for i in range(0, trY.shape[0]):
    if((trY[i,:] == [0,1,0,0,0,0,0,0,0,0]).all()):
        if (new_trX1.size == 0):
            new_trX1 = np.matrix(trX[i,:])
            new_trY1 = np.matrix(trY[i,:])
        else:
            new_trX1 = np.vstack([new_trX1, trX[i,:]])
            new_trY1 = np.vstack([new_trY1, trY[i,:]])
            
for i in range(0, trY.shape[0]):
    if ((trY[i,:] == [0,0,0,0,0,1,0,0,0,0]).all()):
        if (new_trX5.size == 0):
            new_trX5 = np.matrix(trX[i,:])
            new_trY5 = np.matrix(trY[i,:])
        else:
            new_trX5 = np.vstack([new_trX5, trX[i,:]])
            new_trY5 = np.vstack([new_trY5, trY[i,:]])

new_trX1 = np.ceil(new_trX1)
new_trX1 *= 2
new_trX1 -= 1
input_data1 = np.asarray(new_trX1)

new_trX5 = np.ceil(new_trX5)
new_trX5 *= 2
new_trX5 -= 1
input_data5 = np.asarray(new_trX5)

print "Training data formatting done"

#create subset of testing data including only 1's and 5's
new_teX = np.matrix([[]])
new_teY = np.matrix([[]])

for i in range(0, teY.shape[0]):
    if((teY[i,:] == [0,1,0,0,0,0,0,0,0,0]).all() or 
            (teY[i,:] == [0,0,0,0,0,1,0,0,0,0]).all()):
        if (new_teX.size == 0):
            new_teX = np.matrix(teX[i,:])
            new_teY = np.matrix(teY[i,:])
        else:
            new_teX = np.vstack([new_teX, teX[i,:]])
            new_teY = np.vstack([new_teY, teY[i,:]])

new_teX = np.ceil(new_teX)
new_teX *= 2
new_teX -= 1
testing_data = np.asarray(new_teX)
print "Testing data formatting done"

test = np.array([input_data1[0], input_data5[1]])

network = Network(784)
train(network, test)
print "Network training done"

#run on testing data
results = []
for i in range(0, len(testing_data[:20])):
    results.append(network.run(testing_data[i]))

#format data to display as image
for i in range(0, len(results)):
    results[i].shape = (28,28)

start = []
for i in range(0, len(results)):
    temp = testing_data[i]
    temp.shape = (28,28)
    start.append(temp)

#show some results
display(start, results)
