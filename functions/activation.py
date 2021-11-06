import numpy as np


def sigmoid(z: np.ndarray, is_grad: bool=False) -> np.ndarray:
  """
  ## Sigmoid activation function

  ### Arguments:
  
  - z: {np.ndarray or number} value to be evaluated through the sigmoid activation function
  - is_grad: {bool} boolean value which defines if the value to returned is the derivative or not of the evaluation of the sigmoid function
  
  ### Return:
  
  - Evaluated sigmoid function, it depends if is the derivative evaluation on the is_grad variable
  """
  if is_grad:
    s = sigmoid(z, False)
    return s * (1 - s)
  else:
    return 1 / (1 + np.exp(-z))

def tanh(z, is_grad=False):
  """
  ## Hyperbolic tangent activation function
  
  ### Arguments:
  
  - z: {np.ndarray or number} value to be evaluated through the tanh activation function
  - is_grad: {bool} boolean value which defines if the value to returned is the derivative or not of the evaluation of the tanh function
  
  ### Return:
  
  - Evaluated tanh function, it depends if is the derivative evaluation on the is-grad variable
  """
  if is_grad:
    t = tanh(z, False)
    return 1 - (t ** 2)
  else:
    return np.tanh(z)

def relu(z, is_grad=False):
  """
  ## Rectified linear unit activation function
  
  ### Arguments:
  
  - z: {np.ndarray or number} value to be evaluated through the relu activation function
  - is_grad: {bool} boolean value which defines if the value to returned is the derivative or not of the evaluation of the relu function
  
  ### Return:
  
  - Evaluated relu function, it depends if is the derivative evaluation on the is-grad variable
  """
  return np.where(z < 0, 0, 1) if is_grad else np.maximum(0, z)

def leaky_relu(z, is_grad=False):
  """
  ## Leaky rectified linear unit activation function
  
  ### Arguments:
  
  - z: {np.ndarray or number} value to be evaluated through the leaky relu activation function
  - is_grad: {bool} boolean value which defines if the value to returned is the derivative or not of the evaluation of the leaky relu function
  
  ### Return:
  
  - Evaluated leaky relu function, it depends if is the derivative evaluation on the is-grad variable
  """
  return np.where(z < 0, 0.01, 1) if is_grad else np.maximum(0.01 * z, z)

def linear(z, is_grad=False):
  """
  ## Linear activation function
  
  ### Arguments:
  
  - z: {np.ndarray or number} value to be evaluated through the linear activation function
  - is_grad: {bool} boolean value which defines if the value to returned is the derivative or not of the evaluation of the linear function
  
  ### Return:
  
  - Evaluated linear function, it depends if is the derivative evaluation on the is-grad variable
  """
  return 1 if is_grad else z

def softmax(z: np.ndarray) -> np.ndarray:
  """
  ## Softmax activation function
  
  ### Arguments:
  
  - z: {ndarray or number} value to be evaluated through the softmax activation function
  
  Return:
  
  - Evaluated softmax function
  """
  t = np.exp(z)
  return t / np.sum(t)
