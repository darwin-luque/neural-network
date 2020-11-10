from models import ArtificialNeuralNetwork
import optimizers
import utils

from matplotlib import pyplot as plt
import numpy as np

while True:
    try:
        ex_no = input('Ex: ')
        file_no = input('File: ')
        file_handler = open('ex' + ex_no + 'data' + file_no + '.txt')
        break
    except:
        pass

X, Y = [], []

for m, line in enumerate(file_handler):
    values = line.rstrip().split(',')
    X.append([float(v) for v in values[:-1]])
    Y.append(float(values[-1]))

X = np.array(X).reshape(m + 1, -1).T
Y = np.array(Y).reshape(m + 1, -1).T

X_norm = utils.batch_normalization(X)

while True:
    loss = input('Loss type (to watch list of loss functions enter list): ')
    if loss in ['mse', 'bce', 'ce']: break
    elif loss == 'list': print('Valid Loss Functions: mse, bce, ce')
model = ArtificialNeuralNetwork(X.shape[0], loss)
while True:
    add_layer = input('Do you want to add a new layer? (y/n) ')
    if add_layer == 'n': break
    elif add_layer == 'y':
        while True:
            n_units = input('How many units you want in your layer? ')
            try:
                n_units = int(n_units)
                break
            except:
                print('Invalid input! Try again!')
        while True:
            activation_function = input('Select desired activation function (to watch list of functions enter list) ')
            if activation_function == 'list':
                print('Valid Activation Functions: sigmoid, tanh, relu, leaky_relu, linear, softmax')
                continue
            elif activation_function in ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'linear', 'softmax']: break
        while True:
            add_dropout = input('Do you want to add dropout regularization? (y/n) ')
            if add_dropout in ['y', 'n']: break
        if add_dropout == 'y':
            while True:
                keep_prob = input('What probability do you want to keep the unit? (Number less than 1 greater than 0) ')
                try:        
                    keep_prob = float(keep_prob)
                    break
                except:
                    print('Invalid input! Try again!')
                    continue
        else:
            keep_prob = 1
    else:
        print('Ivalid input! Try again!')
        continue
    model.push_layer(n_units, activation_function, keep_prob)

metrics = []
while True:
    metric = input('Enter a desired metric: ')
    if metric.lower() == 'done' or metric == '': break
    metrics.append(metric)

while True:
    epochs = input('Amount of epochs desired: ')
    try:
        epochs = int(epochs)
        break
    except:
        continue

while True:
    learning_rate = input('Learning rate desired: ')
    try:
        learning_rate = float(learning_rate)
        break
    except:
        continue

while True:
    plot = input('Plot the data? (y/n): ')
    if plot == 'y':
        plot = True
        break
    elif plot == 'n':
        plot = False
        break

if plot: plt.scatter(X, Y) 
plt.show()

model, history = optimizers.Adam(X_norm, Y, model, epochs, 8, learning_decay_rate_type='exponential', alpha_0=learning_rate, metrics=metrics, show=True)

plt.plot(np.array(range(1, 1 + epochs)), history['loss'])
plt.title('Loss')
plt.xlabel('epochs')
plt.show()