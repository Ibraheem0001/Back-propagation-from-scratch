import torch
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define the number of input neurons, hidden neurons, and output neurons
input_neurons = 2
hidden_neurons = 5
output_neurons = 1

# Define the learning rate and number of epochs
learning_rate = 0.01
epochs = 10000

# Define the training data
data, label = ds.make_circles(n_samples=1000, factor=.4, noise=0.05)
# data = np.load("dataset/trainX.npy")
# label = np.load("dataset/trainY.npy")
X = torch.from_numpy(data[:1000]).float() # training data point
Y = torch.from_numpy(label[:1000]).float().unsqueeze(dim=1)

# Define the neural network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(input_neurons, hidden_neurons),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_neurons, output_neurons),
    torch.nn.Sigmoid()
)

# Define the loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_loss = []
# Train the neural network
for epoch in range(epochs):
    
    # Forward pass
    y_pred = model(X)
    
    # Compute the loss
    loss = criterion(y_pred, Y)
    train_loss.append(loss.detach().numpy())
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(train_loss)
print("the end")