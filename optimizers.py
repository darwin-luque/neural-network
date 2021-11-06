import numpy as np
from utils import progress_bar

def Gradient_Descent(X, y, model, epochs, batch_size, alpha_0=1e-2, learning_decay_rate_type=None, discrete_reduction=0.001, decay_rate=1, k=1, metrics=[], show=False):
  """
  ## GRADIENT DESCENT

  Gradient Descent optimization algorithm, it can be iterpreted as a Batch, Mini-Batch or Stochastic Gradient Descent depending on the value set in the batch-size variable.\n
  
  ### Arguments:
  
  - X: {Array-like} with the input values of the examples.
  - Y: {Array-like} with the output values of the examples.
  - model: {Model} Object model of the neural network which parameters wants to be optimized.
  - epochs: {int} value that defines iterations desired to run the algorithm through all the batches.
  - batch-size: {int} value that defines batch size in which the examples will be divide to pass through the algorithm.
  - alpha-0: {float} value that defines the initial value of the learning rate.
  - learning-decay-rate-type: {str} value that chooses the type of decay rate desired for the algorithm, this can be:
    1. 'inverse-radical'
    2. 'exponential'
    3. 'inverse'
    4. 'discrete'
    5. if pass any diferent value default alpha-0 set as the learning rate
  - discrete-reduction: {float} If learning-decay-rate-type is chose to be 'discrete', float value to reduce the value of the learning rate at the beginning of each epoch.
  - decay-rate: {float} If learning-decay-rate-type is chose to be 'inverse', float value that defines the decay rate that redices the learning rate each epoch.
  - k: {float} If learning-decay-rate-type is chose to be 'inverse-radical', float value that defines inversely how fast radicaly the learning decays in each epoch.
  - metrics: {str[]} List of strings that define the metrics that wants to be tracked through the optimization algorithm, for now it can only be 'loss' or 'acc'
  - show: {bool} Defines if the metrics and loading bar for the batches are shown in screen
  
  ### Returns:

  - model: {Model} Modified model after all the epochs have been runned.
  - J-history: {float[]} The recording values of all the losses evaluated in each epoch after running through all the batches.
  - acc-history: {float[]} The recording values of all the accuracy evaluated in each epoch after running through all the batches.
  
  * Note: This last 2 are return as a dictionary
  * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
  """
  if X.shape[1] != y.shape[1]:
    raise Exception('Different example size for X and y')
  
  J_history, acc_history = [], []
  
  for i in range(epochs):
    print('Epoch {}/{}'.format(i+1, epochs))
    
    if learning_decay_rate_type == 'inverse_radical': 
      alpha = (k / np.sqrt(i)) * alpha_0
    elif learning_decay_rate_type == 'exponential':
      alpha = (0.95 ** i) * alpha_0
    elif learning_decay_rate_type == 'inverse':
      alpha = (1 / (1 + decay_rate * i)) * alpha_0
    elif learning_decay_rate_type == 'discrete':
      alpha = alpha_0 - discrete_reduction
    else:
      alpha = alpha_0
    
    for t in range(X.shape[1] // batch_size):
      X_batch, y_batch = X[:, t*batch_size:(t+1)*batch_size], y[:, t*batch_size:(t+1)*batch_size]
      
      J, grads = model.cost_function(X_batch, y_batch)
      
      dW, dB = grads
      
      W, B = model.get_parameters()
      W, B = W - alpha * dW, B - alpha * dB
      
      model.set_parameters(W, B)
      
      if show:
        progress_bar(t+1, X.shape[1] // batch_size, length=30)
    
    if 'loss' in metrics and show == True:
      J, _ = model.cost_function(X, y)
    
      print('Training Loss:', J, end='\t')
    
      J_history.append(J)
    
    if ('accuracy' in metrics or 'acc' in metrics) and show == True: 
      if model.loss == 'bce' or model.loss == 'ce':
        acc = np.mean(model.forward_propagation(X)[0][-1] == y)
      else:
        pass
    
      print('Training accuracy:', acc, end='\t')
    
      acc_history.append(acc)
    
    print()
  return model, {'loss': J_history, 'acc': acc_history}

def Adam(X, y, model, epochs, batch_size, alpha_0=1e-2, learning_decay_rate_type=None, discrete_reduction=0.001, decay_rate=1, k=1, beta1=0.9, beta2=0.999, momentum=0, epsilon=1e-8, metrics=[], show=False):
  """
  ## ADAM
  
  Adam optimization algorithm, is an algorithm for first-order gradient.based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.

  For further reference check the paper => [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)
  
  ### Arguments:
  
  - X: {Array-like} with the input values of the examples.
  - Y: {Array-like} with the output values of the examples.
  - model: {Model} Object model of the neural network which parameters wants to be optimized.
  - epochs: {int} value that defines iterations desired to run the algorithm through all the batches.
  - batch-size: {int} value that defines batch size in which the examples will be divide to pass through the algorithm.
  - alpha-0: {float} value that defines the initial value of the learning rate.
  - learning-decay-rate-type: {str} value that chooses the type of decay rate desired for the algorithm, this can be:
    1. 'inverse-radical'
    2. 'exponential'
    3. 'inverse'
    4. 'discrete'
    5. if pass any diferent value default alpha-0 set as the learning rate
  - discrete-reduction: {float} If learning-decay-rate-type is chose to be 'discrete', float value to reduce the value of the learning rate at the beginning of each epoch.
  - decay-rate: {float} If learning-decay-rate-type is chose to be 'inverse', float value that defines the decay rate that redices the learning rate each epoch.
  - k: {float} If learning-decay-rate-type is chose to be 'inverse-radical', float value that defines inversely how fast radicaly the learning decays in each epoch.
  - beta1: {float} value that controls the exponentially weighted average of the momentum
  - beta2: {float} value that controls the exponentially weighted average of the RMSProp
  - momentum: {float} value that defines the starting vector for momentum and RMSprop
  - epsilon: {float} Value to control, when applying RMSProp, no to divide by 0
  - metrics: {string[]} List of strings that define the metrics that wants to be tracked through the optimization algorithm, for now it can only be 'loss' or 'acc'
  - show: {bool} values that defines if the metrics and loading bar for the batches are shown in screen
  
  ### Returns:

  - model: {Model} Modified model after all the epochs have been runned.
  - J-history: {float[]} The recording values of all the losses evaluated in each epoch after running through all the batches.
  - acc-history: {float[]} The recording values of all the accuracy evaluated in each epoch after running through all the batches.
  
  * Note: This last 2 are return as a dictionary
  * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
  """
  if X.shape[1] != y.shape[1]:
    raise Exception('Different example size for X and y')
  
  J_history, acc_history = [], []
  
  for i in range(epochs):
    print('Epoch {}/{}'.format(i+1, epochs))
    
    if learning_decay_rate_type == 'inverse_radical': 
      alpha = (k / np.sqrt(i)) * alpha_0
    elif learning_decay_rate_type == 'exponential':
      alpha = (0.95 ** i) * alpha_0
    elif learning_decay_rate_type == 'inverse':
      alpha = (1 / (1 + decay_rate * i)) * alpha_0
    elif learning_decay_rate_type == 'discrete':
      alpha = alpha_0 - discrete_reduction
    else:
      alpha = alpha_0
      
    V_dW, V_db = np.ones((np.sum([w.size for i, w in enumerate(model.W) if i != 0],))) * momentum, np.ones((np.sum([b.size for j, b in enumerate(model.b) if j != 0],))) * momentum
    S_dW, S_db = np.ones((np.sum([w.size for i, w in enumerate(model.W) if i != 0],))) * momentum, np.ones((np.sum([b.size for j, b in enumerate(model.b) if j != 0],))) * momentum
    
    for t in range(X.shape[1] // batch_size):
      X_batch, y_batch = X[:, t*batch_size:(t+1)*batch_size], y[:, t*batch_size:(t+1)*batch_size]
      
      J, grads = model.cost_function(X_batch, y_batch)
      
      dW, dB = grads
      
      V_dW, V_db = beta1 * V_dW + (1 - beta1) * dW, beta1 * V_db + (1 - beta1) * dB
      S_dW, S_db = beta2 * S_dW + (1 - beta2) * (dW ** 2), beta2 * S_db + (1 - beta2) * (dB ** 2)
      
      V_dW_corrected, V_db_corrected = V_dW / (1 - (beta1 ** t)), V_db / (1 - (beta1 ** t))
      S_dW_corrected, S_db_corrected = S_dW / (1 - (beta2 ** t)), S_db / (1 - (beta2 ** t))
      
      W, B = model.get_parameters()
      
      # W, B = W - alpha * (V_dW_corrected/(np.sqrt(S_dW_corrected) + epsilon)), B - alpha * (V_db_corrected/(np.sqrt(S_db_corrected) + epsilon))
      
      W, B = W - alpha * (V_dW/(np.sqrt(S_dW) + epsilon)), B - alpha * (V_db/(np.sqrt(S_db) + epsilon))
      
      model.set_parameters(W, B)
      
      if show == True:
        progress_bar(t+1, X.shape[1] // batch_size, length=30)
    
    if 'loss' in metrics and show == True:
      J, _ = model.cost_function(X, y)
      
      print('Training Loss:', J, end='\t')
      
      J_history.append(J)
    
    if ('accuracy' in metrics or 'acc' in metrics) and show == True: 
      if model.loss == 'bce' or model.loss == 'ce':
        acc = np.mean(model.forward_propagation(X)[0][-1] == y)
      else:
        pass
      
      print('Training accuracy:', acc, end='\t')
      
      acc_history.append(acc)
    
    print()
  
  return model, {'loss': J_history, 'acc': acc_history}

def Momentum(X, y, model, epochs, batch_size, alpha_0=1e-2, learning_decay_rate_type=None, discrete_reduction=0.001, decay_rate=1, k=1, beta=0.999, momentum=0, metrics=[], show=False):
    """
    ## MOMENTUM

    TODO: DEFINE MOMENTUM OPTIMIZATION FUNC
    
    ### Arguments:
    
    - X: {Array-like} with the input values of the examples.
    - Y: {Array-like} with the output values of the examples.
    - model: {Model} Object model of the neural network which parameters wants to be optimized.
    - epochs: {int} value that defines iterations desired to run the algorithm through all the batches.
    - batch-size: {int} value that defines batch size in which the examples will be divide to pass through the algorithm.
    - alpha-0: {float} value that defines the initial value of the learning rate.
    - learning-decay-rate-type: {string} value that chooses the type of decay rate desired for the algorithm, this can be:
      1. 'inverse-radical'
      2. 'exponential'
      3. 'inverse'
      4. 'discrete'
      5. if pass any diferent value default alpha-0 set as the learning rate
    - discrete-reduction: {float} If learning-decay-rate-type is chose to be 'discrete', float value to reduce the value of the learning rate at the beginning of each epoch.
    - decay-rate: {float} If learning-decay-rate-type is chose to be 'inverse', float value that defines the decay rate that redices the learning rate each epoch.
    - k: {float} If learning-decay-rate-type is chose to be 'inverse-radical', float value that defines inversely how fast radicaly the learning decays in each epoch.
    - beta: {float} value that controls the exponentially weighted average of the momentum
    - momentum: {float} value that defines the starting vector for momentum
    - metrics: {string[]} List of strings that define the metrics that wants to be tracked through the optimization algorithm, for now it can only be 'loss' or 'acc'
    - show: {bool} value that defines if the metrics and loading bar for the batches are shown in screen
    
    ### Returns:

    - model: {Model} Modified model after all the epochs have been runned.
    - J-history: {float[]} The recording values of all the losses evaluated in each epoch after running through all the batches.
    - acc-history: {float[]} The recording values of all the accuracy evaluated in each epoch after running through all the batches.
    
    * Note: This last 2 are return as a dictionary
    * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
    """
    if X.shape[1] != y.shape[1]:
      raise Exception('Different example size for X and y')
    J_history, acc_history = [], []
    for i in range(epochs):
      print('Epoch {}/{}'.format(i+1, epochs))
      if learning_decay_rate_type == 'inverse_radical': 
        alpha = (k / np.sqrt(i)) * alpha_0
      elif learning_decay_rate_type == 'exponential':
        alpha = (0.95 ** i) * alpha_0
      elif learning_decay_rate_type == 'inverse':
        alpha = (1 / (1 + decay_rate * i)) * alpha_0
      elif learning_decay_rate_type == 'discrete':
        alpha = alpha_0 - discrete_reduction
      else:
        alpha = alpha_0
      
      V_dW, V_db = np.ones((np.sum([w.size for i, w in enumerate(model.W) if i != 0],))) * momentum, np.ones((np.sum([b.size for j, b in enumerate(model.b) if j != 0],))) * momentum
      for t in range(X.shape[1] // batch_size):
        X_batch, y_batch = X[:, t*batch_size:(t+1)*batch_size], y[:, t*batch_size:(t+1)*batch_size]
        
        J, grads = model.cost_function(X_batch, y_batch)
        
        dW, dB = grads
        
        V_dW, V_db = beta * V_dW + (1 - beta) * dW, beta * V_db + (1 - beta) * dB
        
        W, B = model.get_parameters()
        W, B = W - alpha * V_dW, B - alpha * V_db
        
        model.set_parameters(W, B)
        
        if show == True:
          progress_bar(t+1, X.shape[1] // batch_size, length=30)
      if 'loss' in metrics and show == True:
        J, _ = model.cost_function(X, y)
        
        print('Training Loss:', J, end='\t')
        
        J_history.append(J)
      if ('accuracy' in metrics or 'acc' in metrics) and show == True: 
        if model.loss == 'bce' or model.loss == 'ce':
          acc = np.mean(model.forward_propagation(X)[0][-1] == y)
        else:
          pass
        
        print('Training accuracy:', acc, end='\t')
        
        acc_history.append(acc)
      print()
    return model, {'loss': J_history, 'acc': acc_history}

def RMS_Prop(X, y, model, epochs, batch_size, alpha_0=1e-2, learning_decay_rate_type=None, discrete_reduction=0.001, decay_rate=1, k=1, beta=0.9, prop=0, epsilon=1e-8, metrics=[], show=False):
    """
    ## RMS PROP
    
    TODO: DEFINE RMS PROP

    ### Arguments:

    - X: {Array-like} with the input values of the examples.
    - Y: {Array-like} with the output values of the examples.
    - model: {Model} Object model of the neural network which parameters wants to be optimized.
    - epochs: {int} value that defines iterations desired to run the algorithm through all the batches.
    - batch-size: {int} value that defines batch size in which the examples will be divide to pass through the algorithm.
    - alpha-0: {float} value that defines the initial value of the learning rate.
    - learning-decay-rate-type: {str} value that chooses the type of decay rate desired for the algorithm, this can be:
      1. 'inverse-radical'
      2. 'exponential'
      3. 'inverse'
      4. 'discrete'
      5. if pass any diferent value default alpha-0 set as the learning rate
    - discrete-reduction: {float} If learning-decay-rate-type is chose to be 'discrete', float value to reduce the value of the learning rate at the beginning of each epoch.
    - decay-rate: {float} If learning-decay-rate-type is chose to be 'inverse', float value that defines the decay rate that redices the learning rate each epoch.
    - k: {float} If learning-decay-rate-type is chose to be 'inverse-radical', float value that defines inversely how fast radicaly the learning decays in each epoch.
    - beta: {float} value that controls the exponentially weighted average of the RMSProp
    - momentum: {float} value that defines the starting vector for RMSProp
    - epsilon: {float} Value to control, when applying RMSProp, no to divide by 0
    - metrics: {str[]} List of strings that define the metrics that wants to be tracked through the optimization algorithm, for now it can only be 'loss' or 'acc'
    - show: {bool} value that defines if the metrics and loading bar for the batches are shown in screen
    
    ### Returns:

    - model: {Model} Modified model after all the epochs have been runned.
    - J-history: {float[]} The recording values of all the losses evaluated in each epoch after running through all the batches.
    - acc-history: {float[]} The recording values of all the accuracy evaluated in each epoch after running through all the batches.
    
    * Note: This last 2 are return as a dictionary
    * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
    """
    if X.shape[1] != y.shape[1]:
      raise Exception('Different example size for X and y')
    
    J_history, acc_history = [], []
    
    for i in range(epochs):
      print('Epoch {}/{}'.format(i+1, epochs))

      if learning_decay_rate_type == 'inverse_radical': 
        alpha = (k / np.sqrt(i)) * alpha_0
      elif learning_decay_rate_type == 'exponential':
        alpha = (0.95 ** i) * alpha_0
      elif learning_decay_rate_type == 'inverse':
        alpha = (1 / (1 + decay_rate * i)) * alpha_0
      elif learning_decay_rate_type == 'discrete':
        alpha = alpha_0 - discrete_reduction
      else:
        alpha = alpha_0
      
      S_dW, S_db = np.ones((np.sum([w.size for i, w in enumerate(model.W) if i != 0],))) * prop, np.ones((np.sum([b.size for j, b in enumerate(model.b) if j != 0],))) * prop
      
      for t in range(X.shape[1] // batch_size):
        X_batch, y_batch = X[:, t*batch_size:(t+1)*batch_size], y[:, t*batch_size:(t+1)*batch_size]
        
        J, grads = model.cost_function(X_batch, y_batch)
        
        dW, dB = grads
        
        S_dW, S_db = beta * S_dW + (1 - beta) * (dW ** 2), beta * S_db + (1 - beta) * (dB ** 2)
        
        W, B = model.get_parameters()
        W, B = W - alpha * (dW/(np.sqrt(S_dW) + epsilon)), B - alpha * (dB/(np.sqrt(S_db) + epsilon))
        
        model.set_parameters(W, B)
        
        if show:
          progress_bar(t+1, X.shape[1] // batch_size, length=30)
      
      if 'loss' in metrics and show == True:
        J, _ = model.cost_function(X, y)
        print('Training Loss:', J, end='\t')
        J_history.append(J)
      
      if ('accuracy' in metrics or 'acc' in metrics) and show == True: 
        if model.loss == 'bce' or model.loss == 'ce':
          acc = np.mean(model.forward_propagation(X)[0][-1] == y)
        else:
          pass
        
        print('Training accuracy:', acc, end='\t')
        
        acc_history.append(acc)
      
      print()
    return model, {'loss': J_history, 'acc': acc_history}