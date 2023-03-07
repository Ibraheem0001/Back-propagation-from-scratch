import torch
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import image as img
import glob

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define the number of input neurons, hidden neurons, and output neurons
input_neurons = 784
hidden1_neurons = 128
hidden2_neurons = 64
output_neurons = 10

# Define the learning rate and number of epochs
learning_rate = 0.01
epochs = 100

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
train_x = np.load('MNIST_dataset_vectors/train_x.npy')
train_y = np.load('MNIST_dataset_vectors/train_y.npy')
test_x = np.load('MNIST_dataset_vectors/test_x.npy')
test_y = np.load('MNIST_dataset_vectors/test_y.npy')


# Define the neural network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(input_neurons, hidden1_neurons),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden1_neurons, hidden2_neurons),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden2_neurons, output_neurons),
    torch.nn.LogSoftmax(dim=1)
)

# Define the loss function and optimizer
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_loss = []
batch_size = 1000
# Train the neural network
for epoch in tqdm(range(epochs)):
    # # create list of random indices that will create our mini-batches
    # idx = [i for i in range(len(train_x))]
    # random.shuffle(idx)
    # iter_data_loader = int(len(train_x)/batch_size)

    # step = 0
    # for i in range(iter_data_loader): # loop over all the batches in one epoch
    #     try:
    #         slice = idx[step:batch_size+step]
    #         step = step + batch_size
    #     except IndexError:  #all indices in one epoch have been traversed
    #         continue
        # Forward pass
        y_pred = model(torch.from_numpy(train_x).reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
        
        # Compute the loss
        loss = criterion(y_pred, torch.from_numpy(train_y))
        train_loss.append(loss.detach().numpy())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(train_loss)
print("the end")