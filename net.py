import numpy as np
from random import shuffle

class Network(object):
    def __init__(self, numIn):
        self.numIn = numIn
        self.weight_matrix = np.random.uniform(-1,1,(numIn,numIn))

    def set_weights(self, weight_matrix):
        self.weight_matrix = weight_matrix
      
    def output(self, neuron, input_data):
        num_neur = self.numIn
        
        w = 0.0
        for i in range(0,num_neur):
            w += self.weight_matrix[neuron][i] * input_data[i]

        active = 1 if w > 0 else -1
        return active 

    def run(self, input_data, max_iter=50):
        count = 0
        result = input_data.copy()

        while True:
            neurons = range(0, self.numIn)
            shuffle(neurons)
            count += 1
            flag = False
            for neuron in neurons:
                output = self.output(neuron, result)

                if output != result[neuron]:
                    result[neuron] = output
                    flag = True

            if not flag or count == max_iter:
                return result

#Training functions
def calc_weight_matrix(neuron, input_data):
    n = len(input_data)
    num_neur = len(input_data[0])
    weights_row = np.zeros(num_neur)     #intialize a weight matrix row
    
    for i in range(0, num_neur):
        if neuron == i:
            continue
        w = 0.0
        for j in range(0, n):
            w += input_data[j][neuron] * input_data[j][i]
        weights_row[i] = (1.0 / float(n)) * w 

    return weights_row

def train(network, input_data):
    num_neur = len(input_data[0])
    weight_matrix = np.zeros((num_neur, num_neur)) #initialize weight matrix
    for i in range(0, num_neur):
        weight_matrix[i] = calc_weight_matrix(i, input_data)

    network.set_weights(weight_matrix)





    





