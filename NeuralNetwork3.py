#this script is a neural network with 2 hidden layers that identifies hand-written numbers. Each hand-written number image is 28 x 28 pixels
#to run in cmd line: python NeuralNetwork3.py train_image.csv train_label.csv test_image.csv
#to run and check answers: python NeuralNetwork3.py train_image.csv train_label.csv test_image.csv test_label.csv

#imports
import math
import numpy as np
import scipy
from scipy.special import expit, logit
import pandas as pd
import sys


# ================================ ALGORITHM ===============================
class NeuralNetwork:
    def __init__(self, learningRate, batchSize, epochs):
        #Learning rate: step size for update weights (e.g. weights = weights - learning * grads)
        self.learningRate = learningRate
        
        #Batch size: number of samples processed each time before the model is updated, 1 <= batch size <= number of samples in training set
        self.batchSize = batchSize
        
        #Number of the epochs: the number of complete passes through the training dataset (e.g. you have 1000 samples, 20 epochs mean you loop this 1000 samples 20 times)
        self.epochs = epochs
        
        #init weights and biases for layers
        #np.random.randn(numRows, numCols) makes a 2d array filled with random values
        #xavier init, weights are randomly within range of: +- sqrt(6)/ (sqrt(incoming networks + outgoing networks))
        self.params = {
            'w1':np.random.randn(512, 784) * (math.sqrt(6) / math.sqrt(784 + 512)),
            'b1':np.random.randn(512, 1) * (math.sqrt(6) / math.sqrt(784 + 512)),
            'w2':np.random.randn(256, 512) * (math.sqrt(6) / math.sqrt(512 + 256)),
            'b2':np.random.randn(256, 1) * (math.sqrt(6) / math.sqrt(512 + 256)),
            'w3':np.random.randn(10, 256) * (math.sqrt(6) / math.sqrt(256 + 10)),
            'b3':np.random.randn(10, 1) * (math.sqrt(6) / math.sqrt(256 + 10))
        }
        #print("w1.shape = " + str(self.params['w1'].shape))
        
    
    # --------------------------------------------------------- METHODS ---------------------------------------------------------
    #forward pass, computes outputs of the layers
    def forwardPass(self, X):
        #cache contains output of each layer
        cache = dict()
        
        #Zn = (weights * inputs) + bias
        #An = activation (sigmoid) of Zn
        #each layer n outputs An
        
        #sigmoid(x) = scipy.special.expit(x)
        
        #1st hidden layer activation
        cache['Z1'] = np.dot(self.params['w1'], X) + self.params['b1']
        cache['A1'] = scipy.special.expit(cache['Z1'])
        
        #2nd hidden layer activation
        cache['Z2'] = np.dot(self.params['w2'], cache['A1']) + self.params['b2']
        cache['A2'] = scipy.special.expit(cache['Z2'])
        
        #last layer (aka output layer) activation
        cache['Z3'] = np.dot(self.params['w3'], cache['A2']) + self.params['b3']
        #cache['A3'] = scipy.special.expit(cache['Z3'])
        cache['A3'] = scipy.special.softmax(cache['Z3'], axis=0)
        
        return cache
        
    
    #backword pass, computes deltas
    def backwardPass(self, X, Y, cache):
        currBatchSize = X.shape[1]
        
        #backprop error = (weight_k * error_j) * transferDeriv(output)
        #error dZ = (prev w * prevError dZ) * sigDer
        #error dZ =      dA            *     sigDer(Z)
        #dw = error dZ * input
        
        #error at last layer = output - expected
        dZ3 = cache['A3'] - Y
        #calculate gradients at last layer
        dw3 = np.dot(dZ3, cache['A2'].T) / currBatchSize
        db3 = np.sum(dZ3, axis=1, keepdims=True) / currBatchSize
        
        #back propogate through last layer to get error
        dA2 = np.dot(self.params['w3'].T, dZ3)
        dZ2 = dA2 * self.sigmoidDeriv(cache['Z2'])
        #calculate gradients of middle layer
        dw2 = np.dot(dZ2, cache['A1'].T) / currBatchSize
        db2 = np.sum(dZ2, axis=1, keepdims=True) / currBatchSize
        
        #back propagate through middle layer
        dA1 = np.dot(self.params['w2'].T, dZ2)
        dZ1 = dA1 * self.sigmoidDeriv(cache['Z1'])
        #calculate gradients of first layer
        dw1 = np.dot(dZ1, X.T) / currBatchSize
        db1 = np.sum(dZ1, axis=1, keepdims=True) / currBatchSize
        
        grads = {
            'dw3':dw3, 
            'db3':db3, 
            'dw2':dw2, 
            'db2':db2, 
            'dw1':dw1, 
            'db1':db1
        }
        return grads
        
    
    #trains the neural network
    def train(self, X, y):
        #repeat for number of epochs
        for i in range(0, self.epochs):
            print("curr epoch = " + str(i))
            #shuffle X and y
            random = np.arange(len(X[1]))
            np.random.shuffle(random)
            #take all rows :, keep random col
            shuffledX = X[:,random]
            shuffledy = y[:,random]
            
            #break X and y into smaller batches
            batches = self.makeBatches(shuffledX, shuffledy, self.batchSize)
            
            for batch in batches:
                #batchx = an image (size 784), batchy = possible outputs (size 10)
                batchx, batchy = batch
                
                #forward pass
                cache = self.forwardPass(batchx)
                
                #compute loss function?
                
                #backward pass, computes gradients
                grads = self.backwardPass(batchx, batchy, cache)
                
                #update weights and biases (weights = weights - learningRate * grads)
                self.params['w1'] = self.params['w1'] - (self.learningRate * grads['dw1'])
                self.params['b1'] = self.params['b1'] - (self.learningRate * grads['db1'])
                self.params['w2'] = self.params['w2'] - (self.learningRate * grads['dw2'])
                self.params['b2'] = self.params['b2'] - (self.learningRate * grads['db2'])
                self.params['w3'] = self.params['w3'] - (self.learningRate * grads['dw3'])
                self.params['b3'] = self.params['b3'] - (self.learningRate * grads['db3'])
                
                
    #make prediction on the labels for the input images (call this after training is done)
    def makePrediction(self, myTestingInput):
        predictionCache = self.forwardPass(myTestingInput)
        output = predictionCache['A3']
        #output: number of cols = number of pictures, number of rows = number of possible outputs (10, answers are 0 - 9). The row index of the biggest number in each col is the label answer
        #np.argmax with axis=0: row index of biggest number in each col
        prediction = np.argmax(output, axis=0)
        return prediction
        
    # ------------------------------- HELPER METHODS -------------------------------------
    #derivative of sigmoid
    def sigmoidDeriv(self, Z):
        sigmoid = scipy.special.expit(Z)
        return sigmoid * (1 - sigmoid)
    
    #splits X (input samples) and y (labels) into smaller batches
    def makeBatches(self, X, y, batchSize):
        m = X.shape[1]
        batches = list()
        totalNumBatches = math.floor(m/batchSize)
        for i in range(0, totalNumBatches):
            batchX = X[:, i * batchSize : (i+1) * batchSize]
            batchy = y[:, i * batchSize : (i+1) * batchSize]
            batch = (batchX, batchy)
            batches.append(batch)
        
        #if the number of input samples cannot neatly be divided by batchSize, the last batch is the remainder
        if m % batchSize != 0:
            batchX = X[:, batchSize * math.floor(m / batchSize) : m]
            batchy = y[:, batchSize * math.floor(m / batchSize) : m]
            batch = (batchX, batchy)
            batches.append(batch)
            
        return batches



# ================================ PROGRAM SETUP ==============================
def main(argv):
    # ------------------------- SETUP ----------------------------
    #read input files (X is inputs for the neural network, Y is outputs of neural network)
    trainingInput = readInput(argv[0])
    trainingAns = readInput(argv[1])
    testingInput = readInput(argv[2])
    
    #convert y (possible values 0 - 9) to one-hot
    yOneHot = np.zeros((trainingAns.size, 10))
    yOneHot[np.arange(trainingAns.size), trainingAns] = 1
    yOneHot = yOneHot.T
    
    
    # ---------------------------- RUN NEURAL NETWORK ----------------------------
    #NeuralNetwork(learningRate, batchSize, epochs)
    myNeuralNetwork = NeuralNetwork(0.02, 100, 20)

    #train network
    print("\nTraining has begun...")
    myNeuralNetwork.train(trainingInput, yOneHot)
    print("Training complete!\n")

    #get predictions on test
    print("Getting predictions...")
    pred = myNeuralNetwork.makePrediction(testingInput)
    #write out the test set predictions
    writeOutput(pred)
    print("Predictions done!\n")
    
    #check answers (for debug purposes)
    testingAns = readInput(argv[3])
    checkPrediction(testingAns, pred)
    
    print("\nPROGRAM DONE RUNNING")
    

#read csv input files
def readInput(filename):
    #read data from file
    print("Reading " + str(filename))
    myInput = pd.read_csv(filename, header=None)
    
    #Transpose, make sure that the file is being read row by row instead of col by col
    myInputFormatted = myInput.T
    
    #use DataFrame.to_numpy() to convert to array
    myInputFormatted = myInputFormatted.to_numpy()
    
    return myInputFormatted

#output prediction results to csv file. param "results" is a numpy array
def writeOutput(results):
    #convert numpy array to DataFrame
    convertedResults = pd.DataFrame(results)
    
    #write to output file
    convertedResults.to_csv("test_predictions.csv", index=None, header=None)

#compare my prediction to the answer (used for debug purposes)
def checkPrediction(answers, predictions):
    #convert answers to proper format
    answers = answers.T
    
    #check how many of my predictions match the answers
    size = len(answers)
    numCorrect = 0
    for i in range(0, size):
        if predictions[i] == answers[i]:
            numCorrect += 1
    percentCorrect = (numCorrect / size) * 100
    print("Percent correct = " + str(percentCorrect) + "%")


# ========================== Python interpreter executes the main method ====================================
if __name__ == "__main__":
   main(sys.argv[1:])