import numpy as np

class Neural_Network:
    def __init__(self, WHO, WHT, WO):
    #---------------Weight Init---------------------------
        #self.WeightInput = WI WI,
        self.WeightHiddenOne = WHO
        self.WeightHiddenTwo = WHT
        self.WeightOutput = WO
    #---------------Bias Init---------------------------
        self.BiasHiddenOne = 0.1
        self.BiasHiddenTwo = 0.1
        self.BiasOutput = 0.1
    #-----------------Weighted input--------------------------
        self.XOne = 0
        self.XTwo = 0
        self.XOut = 0
    #------------Activation layer---------------------
        self.ActivationLayerOne = 0
        self.ActivationLayerTwo = 0

    def sigmoid(self, z):
        return (1.0 / (1 + np.exp(-z)))

    def sigmoid_prime(self, z):
        return (self.sigmoid(z) * (1-self.sigmoid(z)))

    def MSE_cost(self, yHat, y):
        return np.sum((yHat - y)**2) / y.size


    def MSE_cost_prime(self, yHat, y):
        return (yHat - y)

    def feed_forward(self, X):
        #WeightOutput(Sigmond(WeightHiddenTwo(sigmond(WeightHiddenOne(X)))))
        # Hidden layer One
        self.XOne = np.dot(X, self.WeightHiddenOne) + self.BiasHiddenOne
        print("----------------XOne-------------------")
        print(self.XOne)
        self.ActivationLayerOne = self.sigmoid(self.XOne)
        print("----------------ActivationLayerOne-------------------")
        print(self.ActivationLayerOne)
        # Hidden layer Two
        self.XTwo = np.dot(self.ActivationLayerOne, self.WeightHiddenTwo) + self.BiasHiddenTwo
        self.ActivationLayerTwo = self.sigmoid(self.XTwo)
        print("----------------ActivationLayerTwo-------------------")
        print(self.ActivationLayerTwo)


        # Output layer
        self.XOut = np.dot(self.ActivationLayerTwo, self.WeightOutput) + self.BiasOutput
        print("----------------XOut-------------------")

        print(self.XOut)
        
    def backprop(self, X, y, lr):
        
        # Layer Error
        OutputError = self.MSE_cost_prime(self.XOut, y)
        delta3 = OutputError
        print("ActivationLayerTwo.T")
        print(self.ActivationLayerTwo.T)
        djdwo = np.dot(self.ActivationLayerTwo.T, delta3)
        print("djdwo")
        print(djdwo)
        delta2 = np.dot(delta3, self.WeightOutput.T)*self.sigmoid_prime(self.XTwo)
        dJdW2 = np.dot(self.ActivationLayerOne.T, delta2)
        
        delta1 = np.dot(delta2, self.WeightHiddenTwo.T)*self.sigmoid_prime(self.XOne)
        dJdW1 = np.dot(X.T, delta1)  
        # Update weights
        #print("dJdW1", dJdW1.shape)
        #print("dJdW2", dJdW2.shape)
        #print("djdwo", djdwo.shape)
        self.WeightHiddenOne -= lr * dJdW1
        self.WeightHiddenTwo -= lr * dJdW2
        self.WeightOutput -= lr * djdwo


    def printShapes(self):
        print("WeightHiddenOne", self.WeightHiddenOne.shape)
        print("WeightHiddenTwo", self.WeightHiddenTwo.shape)
        print("WeightOutput", self.WeightOutput.shape)
    #---------------Bias Init---------------------------
        print("BiasHiddenOne", self.BiasHiddenOne)
        print("BiasHiddenTwo", self.BiasHiddenTwo)
        print("BiasOutput", self.BiasOutput)
    #-----------------Weighted input--------------------------
        print("XOne", self.XOne.shape)
        print("XTwo", self.XTwo.shape)
        print("XOut", self.XOut.shape)
    #------------Activation layer---------------------
        print("ActivationLayerOne", self.ActivationLayerOne.shape)
        print("ActivationLayerTwo", self.ActivationLayerTwo.shape)

    def SaveToFile(self):
        np.save('WeightHiddenOne.npy', self.WeightHiddenOne)    # .npy extension is added if not given
        np.save('WeightHiddenTwo.npy', self.WeightHiddenTwo)
        np.save('WeightOutput.npy', self.WeightOutput)

    def LoadFromFile(self):
        self.WeightHiddenOne = np.load('WeightHiddenOne.npy')
        self.WeightHiddenTwo = np.load('WeightHiddenTwo.npy')
        self.WeightOutput = np.load('WeightOutput.npy')

    def PrintWeights(self):
        print(self.WeightHiddenOne)
        print("-----------------------------------")
        print(self.WeightHiddenTwo)
        print("-----------------------------------")
        print(self.WeightOutput)
        print("-----------------------------------")
        
