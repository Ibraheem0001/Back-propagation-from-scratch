# Include libraries which may use in implementation
import numpy as np
import random
from sklearn.utils import shuffle
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import pickle


# Create a Neural_Network class
class Neural_Network(object):        
    def __init__(self,inputSize = 2, hiddenlayer = 3, outputSize = 1):        
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer = hiddenlayer
        #weights
        self.W1 = np.random.rand(inputSize +1, hiddenlayer)
        # W1 = np.load('parameters/weight_0.npy')
        # b1 = np.expand_dims(np.load('parameters/bias_0.npy'), axis=1)
        # self.W1 = np.concatenate((W1, b1), axis=1).T
        # size of the wieght will be (inputSize +1, hiddenlayer) that +1 is for bias   
         
        self.W2 = np.random.rand(hiddenlayer +1, outputSize) 
        # W2 = np.load('parameters/weight_2.npy')
        # b2 = np.expand_dims(np.load('parameters/bias_2.npy'), axis=1)
        # self.W2 = np.concatenate((W2, b2), axis=1).T
        # size of the wieght will be (hiddenlayer +1, outputSize) that +1 is for bias    
        
    def feedforward(self, X):
        #forward propagation through our network
        # dot product of X (input) and set of weights
        # apply activation function (i.e. whatever function was passed in initialization) 
        self.X = X 
        self.out1 = self.X@self.W1
        self.out2 = self.sigmoid(self.out1)
        self.out2 = add_bias_column(self.out2)
        self.out3 = self.out2@self.W2
        y_pred = self.sigmoid(self.out3)  
        return y_pred # return your answer with as a final output of the network

    def sigmoid(self, s):
        # apply sigmoid function on s and return it's value
        g = np.zeros(s.shape)
        # split into positive and negative to improve stability
        g[s>=0.0] = 1.0 / (1.0 + np.exp(-s[s>=0.0]))
        g[s<0.0] = np.exp(s[s<0.0]) / (np.exp(s[s<0.0])+1.0)
        return g 

    def sigmoid_derivative(self, s):
        #derivative of sigmoid
        return self.sigmoid(s) * (1-self.sigmoid(s)) # apply derivative of sigmoid on s and return it's value 
    
    def tanh(self, s):
        # activation function
        return 0 # apply tanh function on s and return it's value

    def tanh_derivative(self, s):
        #derivative of tanh
        return 0 # apply derivative of tanh on s and return it's value
    
    def relu(self, s):
        # activation function
        s[s<0]=0
        return s # apply relu function on s and return it's value

    def relu_derivative(self, s):
        #derivative of relu
        return 0 # apply derivative of relu on s and return it's value

    def backwardpropagate(self,X, Y, y_pred, lr):

        # compute error in output which is loss compute cross entropy loss function
        self.loss = self.crossentropy(Y, y_pred)
        # backward propagate through the network
        dy_pred = (-Y/np.squeeze(y_pred) + ((1-Y)/np.squeeze(1-y_pred)))/len(Y)
        d_out3 = self.sigmoid_derivative(self.out3) * np.expand_dims(dy_pred, axis=1)
        dW2 = self.out2 * d_out3
        dW2 = np.sum(dW2, axis=0, keepdims=True).T  # adding the derivatives for all examples in the batch
        d_out2 = self.W2[:-1] * d_out3.T # last value in self.W2 is for bias, which we do not backpropagate further
        d_out1 = self.sigmoid_derivative(self.out1) * d_out2.T
        dW1 = (self.X).T @ d_out1

        # SGD update step
        # self.W1 = self.W1 - lr*dW1
        # self.W2 = self.W2 - lr*dW2

        # use momentum update with SGD for better convergence
        self.z1 = self.beta*self.z1 + dW1
        self.W1 = self.W1 - lr*self.z1 #lr * mu *v - lr * dx
        self.z2 = self.beta*self.z2 + dW2
        self.W2 = self.W2 - lr*self.z2 

    
    def crossentropy(self, Y, Y_pred):
        # compute error based on crossentropy loss 
        return (np.sum(-np.log(Y_pred[Y.astype(bool)])) + np.sum(-np.log(1-Y_pred[(1-Y).astype(bool)])))/len(Y) #error

    def train(self, trainX, trainY, epochs = 100, learningRate = 0.001, plot_err = True ,validationX = None, validationY = None):
        # feed forward trainX and trainY and recivce predicted value
        # backpropagation with trainX, trainY, predicted value and learning rate.
        # if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
        # plot error of the model if plot_err is true
        train_loss = []
        batch_size = 16
        self.z1 = 0
        self.beta = 0.9
        self.z2 = 0
        for epoch in range(epochs):
                y_pred = self.feedforward(trainX)
                self.backwardpropagate(trainX, trainY, y_pred, learningRate)
                train_loss.append(self.loss)
        return train_loss


    def predict(self, testX):
        # predict the value of testX
        y_pred = self.feedforward(testX)
        y_pred[y_pred>0.5]=1
        y_pred[y_pred<=0.5]=0
        return y_pred.squeeze()
    
    def accuracy(self, testX, testY):
        # predict the value of trainX
        # compare it with testY
        # compute accuracy, print it and show in the form of picture
        y_pred = self.predict(testX)
        bool_result = (y_pred==testY)
        bool_result[bool_result==True]=1
        bool_result[bool_result==False]=0
        accuracy = np.sum(bool_result)*100/len(testY)
        return accuracy # return accuracy    
        
    def saveModel(self,name):
        # save your trained model, it is your interpretation how, which and what data you store
        # which you will use later for prediction
        pickle.dump(self, open(name, "wb"))

        
    def loadModel(self,name):
        # load your trained model, load exactly how you stored it.
        pass

# this function appends a column of ones to X (bias column)
def add_bias_column(X):
    bias = np.ones((X.shape[0],1))
    X = np.concatenate((X, bias), axis=1)
    return X

def main():   
    random.seed(0)
    np.random.seed(0)
    data, label = ds.make_circles(n_samples=1000, factor=.4, noise=0.05)

    #Lets visualize the dataset
    reds = label == 0
    blues = label == 1
    # plt.scatter(data[reds, 0], data[reds, 1], c="red", s=20, edgecolor='k')
    # plt.scatter(data[blues, 0], data[blues, 1], c="blue", s=20, edgecolor='k')
    # plt.show()

    # add bias column (column of 1's to the input)
    data = add_bias_column(data)

    #Note: shuffle this dataset before dividing it into three parts
    data, label = shuffle(data, label)

    # Explicitly distributing this data into three parts i.e. training, validation and testing,
    # could also use sklearn's train test split method
    trainX = data[:800] # training data point
    trainY = label[:800] # training lables

    validX = data[800:900] # validation data point
    validY = label[800:900] # validation lables

    testX = data[900:1000] # testing data point
    testY = label[900:1000] # testing lables


    model = Neural_Network(2,5)
    # try different combinations of epochs and learning rate
    train_loss = model.train(trainX, trainY, epochs = 10000, learningRate = 0.01, validationX = validX, validationY = validY)
   
    plt.plot(train_loss)
    #save the best model which you have trained, 
    model.saveModel('bestmodel.mdl')


    # create class object
    mm = Neural_Network()
    # load model which will be provided by you
    mm.loadModel('bestmodel.mdl')
    # check accuracy of that model
    accuracy = model.accuracy(testX,testY)
    print(f"the accuracy of our model on test dataset is {accuracy}%")


if __name__ == '__main__':
    main()