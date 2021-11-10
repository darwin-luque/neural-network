import numpy as np
import matplotlib.pyplot as plt
from models import ArtificialNeuralNetwork
from optimizers import Gradient_Descent
import utils


if __name__ == '__main__':
  while True:
    try:
      ex_no = input('Ex: ')
      file_no = input('File: ')
      file_handler = open('./data/ex' + ex_no + 'data' + file_no + '.txt')
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

  print(X.shape)

  plt.scatter(X, Y)
  plt.show()
  
  X_norm = utils.batch_normalization(X)

  plt.scatter(X_norm, Y)
  plt.show()

  ann = ArtificialNeuralNetwork(X_norm.shape[0], 'mse')

  ann.push_layer(10, 'sigmoid', 1)
  ann.push_layer(5, 'sigmoid', 1)
  ann.push_layer(1, 'linear', 1)
  model: ArtificialNeuralNetwork
  model, metrics  = Gradient_Descent(X_norm, Y, ann, 500, X.shape[1], 0.01, metrics=['acc', 'loss'], show=True)

  plt.scatter(X, Y)
  print(X)
  x_min, x_max = X.reshape(-1).min(), X.reshape(-1).max()
  testing_X = np.arange(x_min, x_max + 1, 0.1).reshape(X.shape[0], -1)
  preds = model.prediction(testing_X)
  plt.scatter(testing_X, preds, color='red')
  plt.show()
