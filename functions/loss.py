import numpy as np


def mse(pred: np.ndarray, Y: np.ndarray, m: int) -> float:
  """
  ## MEAN SQUARED ERROR LOSS FUNCTION

  ### Arguments:
  
  - pred: {np.ndarray} predictions based on the input values of the examples
  - Y: {np.ndarray} output values of the examples
  - m: {int} number of examples
  
  ### Return:
  
  - Evaluated mean squared error loss function in all the examples between the predicted value given by the variable X and the variable Y
  """
  # A, _ = forward_propagation(X)
  dif_Y = pred - Y
  return (dif_Y @ dif_Y.T).squeeze().squeeze() / m

def bce(pred: np.ndarray, Y: np.ndarray, m: int) -> float:
  """
  ## BINARY CROSS ENTROPY LOSS FUNCTION
  
  ### Arguments:
  
  - pred: {np.ndarray} predictions based on the input values of the examples
  - Y: {np.ndarray} output values of the examples
  - m: {int} number of examples
  
  ### Return:
  
  - Evaluated log-like loss function in all the examples for binary class discrete models between the predicted value given by the variable X and the variable Y
  """
  # A, _ = self.forward_propagation(X)
  return - (Y @ np.log(pred).T + (1 - Y) @ np.log(1 - pred).T).squeeze().squeeze() / m

def ce(pred: np.ndarray, Y: np.ndarray, m: int):
  """
  ## CROSS ENTROPY LOSS FUNCTION
  
  ### Arguments:
  
  - pred: {np.ndarray} predictions based on the input values of the examples
  - Y: {np.ndarray} output values of the examples
  - m: {int} number of examples
  
  ### Return:
  
  - Evaluated log-like loss function in all the examples between for multi class discrete models the predicted value given by the variable X and the variable Y
  """
  # A, _ = self.forward_propagation(X)
  return - np.sum(np.log(pred) @ Y.T) / m
