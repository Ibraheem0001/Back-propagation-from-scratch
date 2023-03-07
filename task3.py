# Include libraries which may use in implementation
import numpy as np
import random
from sklearn.utils import shuffle
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import pickle
from matplotlib import image as img
import glob
from tqdm import tqdm

# Create a Neural_Network class
class Neural_Network(object):        
    def __init__(self,inputSize = 784, hiddenlayer1 = 128, hiddenlayer2 = 64, outputSize = 10):        
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer1 = hiddenlayer1
        self.hiddenLayer2 = hiddenlayer2

        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.beta = 0.9
        
        #weights
        # self.W1 = np.random.rand(inputSize +1, hiddenlayer1)
        # self.W2 = np.random.rand(hiddenlayer1 +1, hiddenlayer2)        
        # self.W3 = np.random.rand(hiddenlayer2 +1, outputSize)   

        W1 = np.load('parameters/weight_0.npy')
        b1 = np.expand_dims(np.load('parameters/bias_0.npy'), axis=1)
        self.W1 = np.concatenate((W1, b1), axis=1).T
        W2 = np.load('parameters/weight_2.npy')
        b2 = np.expand_dims(np.load('parameters/bias_2.npy'), axis=1)
        self.W2 = np.concatenate((W2, b2), axis=1).T
        W3 = np.load('parameters/weight_4.npy')
        b3 = np.expand_dims(np.load('parameters/bias_4.npy'), axis=1)
        self.W3 = np.concatenate((W3, b3), axis=1).T
        
    def feedforward(self, X):
        #forward propagation through our network
        # dot product of X (input) and set of weights
        # apply activation function (i.e. whatever function was passed in initialization) 
        self.X = X 
        self.out1 = self.X@self.W1
        self.out2 = self.sigmoid(self.out1)
        self.out2 = add_bias_column(self.out2)
        self.out3 = self.out2@self.W2
        self.out4= self.sigmoid(self.out3)  
        self.out4 = add_bias_column(self.out4)
        self.out5 = self.out4@self.W3
        y_pred= self.softmax(self.out5)  
        return y_pred # return your answer with as a final output of the network

    def softmax(self, s):
        s = np.exp(s)
        sum = np.sum(s, axis=1, keepdims=True)
        s = s/sum
        return s
    
    def softmax_derivative(self, s):
        softmax = self.softmax(s)
        # below we compute -softmax * softmax, but we use broadcasting to get a matrix
        # where each (i,j)th element = -softmax[i]*[j]
        temp1 = np.expand_dims(softmax, axis=2)
        temp2 = np.expand_dims(softmax, axis=1)
        derivative = (-temp1*temp2)

        # create a boolean mask of diagonal elements for all batch examples
        diag_mask = np.eye(derivative.shape[1], dtype=bool)[np.newaxis,:,:].repeat(derivative.shape[0], axis=0)
        
        # diagonal elements' derivative
        diag = softmax * (1 - softmax)

        # fill the diagonal of the derivative with diagonal values
        derivative[diag_mask] = diag.flatten()

        #Replaced below code with above numpy broadcasting
        # a = np.zeros((10,10))
        # np.fill_diagonal(a, diag)
        # for i in range(10):
        #     for j in range(10):
        #         if i != j:
        #             a[i][j] = -softmax[0][i]*softmax[0][j]
        return derivative

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
        self.loss = (self.crossentropy(Y, y_pred))/len(Y)
        # backward propagate through the network
        dy_pred = np.zeros_like(y_pred)
        dy_pred[np.arange(len(Y)), Y] = (-1/y_pred[np.arange(len(Y)), Y])/len(Y)
        d_out5 = self.softmax_derivative(self.out5) * np.expand_dims(dy_pred, axis=2)
        d_out5 = np.sum(d_out5,axis=1)
        dW3 = self.out4.T @ d_out5
        d_out4 = self.W3[:-1] @ d_out5.T # last value in self.W3 is for bias, which we do not backpropagate further
        d_out3 = self.sigmoid_derivative(self.out3) * d_out4.T
        dW2 = self.out2.T @ d_out3
        d_out2 = self.W2[:-1] @ d_out3.T # last value in self.W2 is for bias, which we do not backpropagate further
        d_out1 = self.sigmoid_derivative(self.out1) * d_out2.T
        dW1 = (self.X).T @ d_out1

        # SGD update step
        # self.W1 = self.W1 - lr*dW1
        # self.W2 = self.W2 - lr*dW2
        # self.W3 = self.W3 - lr*dW3

        # use momentum update with SGD for better convergence
        self.z1 = self.beta*self.z1 + dW1
        self.W1 = self.W1 - lr*self.z1 #lr * mu *v - lr * dx
        self.z2 = self.beta*self.z2 + dW2
        self.W2 = self.W2 - lr*self.z2 
        self.z3 = self.beta*self.z3 + dW3
        self.W3 = self.W3 - lr*self.z3 

    
    def crossentropy(self, Y, Y_pred):
        # compute error based on crossentropy loss 
        return -np.sum(np.log(Y_pred[np.arange(len(Y)), Y])) #error

    def train(self, trainX, trainY, epochs = 100, learningRate = 0.001, plot_err = True ,validationX = None, validationY = None):
        # feed forward trainX and trainY and recivce predicted value
        # backpropagation with trainX, trainY, predicted value and learning rate.
        # if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
        # plot error of the model if plot_err is true
        train_loss = []
        batch_size = 16

        for epoch in tqdm(range(epochs)):
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
    # Define the training data
    def load_dataset(path):
        train_x, train_y, test_x, test_y = [], [], [], []
        for i in range(10):
            for filename in glob.glob(path + "/train/" + str(i) + "/*.png"):
                im=img.imread(filename)
                train_x.append(im)
                train_y.append(i)
        for i in range(10):
            for filename in glob.glob(path + "/test/" + str(i) + "/*.png"):
                im=img.imread(filename)
                test_x.append(im)
                test_y.append(i)
                return np.array(train_x), np.array(train_y), np.array(test_x),np.array(test_y)

    # train_x, train_y, test_x, test_y =load_dataset("Task3_MNIST_Data")
    trainX = np.load('MNIST_dataset_vectors/train_x.npy')
    trainY = np.load('MNIST_dataset_vectors/train_y.npy')
    testX = np.load('MNIST_dataset_vectors/test_x.npy')
    testY = np.load('MNIST_dataset_vectors/test_y.npy')

    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2])
    testX = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2]) 
    # add bias column (column of 1's to the input)
    trainX = add_bias_column(trainX)
    testX = add_bias_column(testX)


    model = Neural_Network(784, 128, 64, 10)
    # try different combinations of epochs and learning rate
    train_loss = model.train(trainX, trainY, epochs = 100, learningRate = 0.01)
   
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