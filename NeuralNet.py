import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice
import math

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        """
            Sigmoid Function: S(value) = 1 / (1 + e^(-value))
        """
        return 1/(1 + math.exp(-value))

      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as rounded g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The rounded value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        inActs.insert(0, 1.0) #this is the bias input added to the front of activationList
        weightedSum = self.getWeightedSum(inActs) #helper method to compute overall weightedSum
        del inActs[0]  #getting rid of bias from the list
        return round(self.sigmoid(weightedSum)) #calling sigmoid fucntion on it as helper method



    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        return math.exp(value)/(math.pow((math.exp(value) + 1),2))
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        inActs.insert(0, 1.0)
        weightedSum = self.getWeightedSum(inActs)
        del inActs[0]
        return self.sigmoidDeriv(weightedSum)
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (absolute total)
        """
        totalModification = 0
        """YOUR CODE"""
        inActs.insert(0, 1.0)
        index = 0
        weightList = []
        for weight in self.weights:
            oldWeight = weight
            weight = weight + (alpha * inActs[index] * delta) #updating weight
            weightList.append(weight)
            diff = abs(weight - oldWeight) #calculating difference between them
            totalModification += diff
            index += 1

        del inActs[0]
        self.weights = weightList
        return totalModification
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values 0f all perceptrons in each layer.
        """
        """YOUR CODE"""
        returnList = []
        returnList.append(inActs)
        currentInput = inActs
        for currLayer in self.layers:
            pList = []
            for perceptor in currLayer:
                pList.append(perceptor.sigmoidActivation(currentInput))

            currentInput = pList #updating the currentInput List after each layer
            returnList.append(pList)

        return returnList


    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        NOTE : the pseudo code in the book has an error - 
        you should not update the weights while backpropagating; 
        follow the comments below or the description in lecture.
        
        Args: 
            examples (list<tuple<list,list>>):for each tuple first element is input(feature) "vector" (list)
                                                             second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed weight change of all perceptrons, divided by the sum of 
               their input sizes.
        """
        #keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0
        avgError = 0
        totalWeightChange = 0

        for example in examples:#for each example
            deltas = []#keep track of deltas to use in weight change
            """YOUR CODE"""
            """Get output of all layers"""
            output = self.feedForward(example[0])
            """
            Calculate output errors for each output perceptron and keep track 
            of error sum. Add error delta values to list.
            """

            index = 0
            deltaOutputList = []
            while index < len(self.outputLayer):
                avgError += ((example[1][index] - output[-1][index]) ** 2) / 2
                deltaOutputList.append(self.outputLayer[index].sigmoidActivationDeriv(output[-2]) * (example[1][index] - output[-1][index]))
                index += 1
            deltas.append(deltaOutputList)
            
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            """
            index = len(self.layers) - 2
            while index >= 0:
                deltaCurrLayer = []
                percepIndex = 0
                #need to calculate weightedSum for all layers above the current one
                for currentPercept in self.layers[index]:
                    weightedSum = 0
                    deltaJIndex = 0
                    targetPercep = self.layers[index+1]
                    for abovePercept in targetPercep:
                        weightedSum += deltas[0][deltaJIndex] * abovePercept.weights[percepIndex+1]  #weightedSum
                        deltaJIndex += 1

                    deltaCurrPerceptron = weightedSum * currentPercept.sigmoidActivationDeriv(output[index])  #calculating deltai here

                    deltaCurrLayer.append(deltaCurrPerceptron)
                    percepIndex += 1

                index -= 1
                deltas.insert(0, deltaCurrLayer)
            """
            Having aggregated all deltas, update the weights of the
            hidden and output layers accordingly.
            """
            index = 0
            for layer in self.layers:
                percepIndex = 0
                for perceptron in layer:
                    totalWeightChange += perceptron.updateWeights(output[index], alpha, deltas[index][percepIndex])
                    numWeights += len(perceptron.weights)
                    percepIndex += 1
                index += 1

        #end for each example
        
        """Calculate final output"""
        averageError = (avgError*1.0) / ((len(examples)*len(self.outputLayer)))
        averageWeightChange = (totalWeightChange*1.0) / numWeights
        return averageError, averageWeightChange
    
def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """

    #Set alpha to 0 hopefully to not change the weights will change the network
    examplesTrain,examplesTest = examples
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet

    """
    YOUR CODE
    """
    #examplesTrain is a tuple of list of input and output that you need to train your network, basically passed in as
    #"examples" that we used to train our network in backPropogation
    iteration = 0
    trainError = 0
    weightMod = 0
    trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)
    iteration += 1
    while (weightMod > weightChangeThreshold) & (iteration < maxItr):
        trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)
        iteration += 1

    """
    Iterate for as long as it takes to reach weight modification threshold
    """
        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',
        
          
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
    """ 
    
    testError = 0
    testGood = 0
    testAccuracy = 0 #num correct/num total

    #examplesTest is a tuple of list to compare your trained networks input and output against
    #these preset determined test values
    for test in examplesTest:
        fullOutput = nnet.feedForward(test[0])
        LastLayerOutput = fullOutput[len(fullOutput)-1]
        if LastLayerOutput == test[1]:  #test[1] is a list of output that is testing your trained network results against it
            testGood += 1
        else:
            testError += 1
    total = testGood + testError
    testAccuracy = (testGood * 1.0)/total


    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    
    """return something"""
    return nnet, testAccuracy

