import numpy as np

x_enter = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]), dtype=float)
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)  # Output data 1 = red / 0 = blue

x_enter = x_enter / np.amax(x_enter, axis=0)

x = np.split(x_enter, [8])[0]
xPrediction = np.split(x_enter, [8])[1]


def sigmoid(_s):
    return 1 / (1 + np.exp(- _s))


class NeuralNetwork(object):

    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # Matrices 2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # Matrices 3x1

    def sigmoid_prime(self, s):
        return s * (1 - s)

    def forward(self, _x):
        self.z = np.dot(_x, self.W1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = sigmoid(self.z3)

        return o

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)


        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        _o = self.forward(X)
        self.backward(X, y, _o)

    def predict(self):
        print('Predicted data after training: ')
        print('Input: \n' + str(xPrediction))
        print('Output: \n' + str(self.forward(xPrediction)))

        if self.forward(xPrediction) < 0.5:
            print('Its blue \n')
        else:
            print('Its red \n')


NN = NeuralNetwork()

for i in range(70000):
    print('# ' + str(i), '\n')
    print('Input value: \n', str(x))
    print('Actual output: \n', str(y))
    print('Output predicted: \n', str(np.matrix.round(NN.forward(x), 2)))
    print('\n')
    NN.train(x, y)

NN.predict()
