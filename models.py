import numpy as np

class ArtificialNeuralNetwork:
  def __init__(self, no_inputs: int, loss: str):
    """
    ## Artificial Neural Network
    
    ### Arguments:
    
    - no-inputs: {int} Number inputs that the model will be receiving, the inputs should have a shape (n, m) where n is the value to set and m is the number of examples
    - loss => The type of loss function to be used, can be:
        1. Mean Squared Error (pass in: 'mse')
        2. Binary Cross Entropy (pass in: 'bce')
        3. Cross Entropy (pass in: 'ce')
    """
    self.layers_sizes = [no_inputs]
    self.loss = loss
    self.W, self.b, self.act_funcs, self.keep_probs = [0], [0], ['linear'], [1]
  
  def xavier_initialization(self, n0: int, n1: int):
    """
    ## XAVIER INITIALIZATION

    Random initialization for parameters of deep neural network scaled depending on the size of the layers that surround the parameters\n
    
    ### Arguments:
    
    - n0: {int} shape of the layer that comes before parameters, also the column size of the output array
    - n1: {int} shape of the layer that comes after parameters, also the row size of the output array
    
    ### Return:
    
    - Array with shape (n1, n0)
    """
    return np.random.randn(n1, n0) * ((2 / (n0 + n1))**0.5)
  
  def push_layer(self, n_units: int, activation_function: str, keep_prob=1):
    """
    ## LAYER ADDER
    
    ### Arguments:
    
    - n_units: {int} number of units to be added in the current layer
    - activation-function: {str} desired activation function for the current layer, can be:
        1. Sigmoid (pass in: 'sigmoid')
        2. Hyperbolic Tangent (pass in: 'tanh')
        3. Rectified Linear Unit (pass in: 'relu')
        4. LEAKY REctified Linear Unit (pass in: 'leaky_relu')
    - keep_prob: {float} probability to keep different units of the layer, as form to apply dropout layer
    """
    self.layers_sizes.append(n_units)
    self.W.append(self.xavier_initialization(self.layers_sizes[-2], self.layers_sizes[-1]))
    self.b.append(np.zeros((self.layers_sizes[-1], 1)))
    self.act_funcs.append(activation_function)
    self.keep_probs.append(keep_prob)
  
  def activation_function(self, z: np.ndarray, is_grad: bool, act_func: str):
    """
    ## GENERAL ACTIVATION FUNCTION GETTER FOR THE OBJECT
    
    The objective of this method is to control all the available activation functions
    
    ### Arguments:
    
    - z: {ndarray or number} value to be evaluated through the softmax activation function
    - is_grad: {boolean} value which defines if the value to returned is the derivative or not of the evaluation of the linear function
    - act_func: {string} dictating the desired activation function, can be:
        1. 'sigmoid'
        2. 'tanh'
        3. 'relu'
        4. 'leaky-relu'
        5. 'linear'
        6. 'softmax'
    
    ### Return:
    
    - Evaluated respective activation function given the variable act_func
    """
    if act_func == 'sigmoid': return self.sigmoid(z, is_grad)
    elif act_func == 'tanh': return self.tanh(z, is_grad)
    elif act_func == 'relu': return self.relu(z, is_grad)
    elif act_func == 'leaky_relu': return self.leaky_relu(z, is_grad)
    elif act_func == 'linear': return self.linear(z, is_grad)
    elif act_func == 'softmax': return self.softmax(z)
    else: return 'Error'
  
  def forward_propagation(self, X: np.ndarray):
      """
      ## FORWARD PROPAGATION LOSS FUNCTION

      ### Arguments:
      - X: {np.ndarray} input values of the examples
      
      ### Return:

      - All the evaluated units activated and not activated values, stored in the variables A and Z, respectively.
      """
      A, Z = [X], [None]
      for i in range(1, len(self.layers_sizes)):
          Z.append(self.W[i] @ A[i-1] + self.b[i])
          a = self.activation_function(Z[i], False, self.act_funcs[i])
          d = np.random.rand(a.shape[0], a.shape[1]) < self.keep_probs[i]
          A.append((a * d) / self.keep_probs[i])
      return A, Z

  def backward_propagation(self, X: np.ndarray, Y: np.ndarray):
    """
    ## BACKWARD PROPAGATION LOSS FUNCTION

    ### Arguments:
    
    - X: {np.ndarray} input values of the examples
    - Y: {np.ndarray} output values of the examples
    
    ### Return:
    
    - dW: {np.ndarray} Gradient of the W paramaters
    - db: {np.ndarray} Gradient of the b parameters
    """
    A, Z = self.forward_propagation(X)
    dW, db = [], []
    if self.act_funcs[-1] == 'softmax' or self.act_funcs[-1] == 'sigmoid' or self.loss == 'mse':
        dZ = [A[-1] - Y]
    else:
        dA = np.divide(1-Y, 1-A[-1]) - np.divide(Y, A[-1])
        dZ = [dA * self.activation_function(Z[-1], True, self.act_funcs[-1])]
    for i in reversed(range(len(self.layers_sizes) - 1)):
        dW.append((dZ[0] @ A[i].T) / X.shape[1])
        db.append(np.sum(dZ[0], axis=1, keepdims=True))
        if i == 0: break
        dZ.insert(0, (self.W[i+1].T @ dZ[0]) * self.activation_function(Z[i], True, self.act_funcs[i]))
    dW.append(0)
    db.append(0)
    dW.reverse()
    db.reverse()
    return dW, db
  
  def cost_function(self, X: np.ndarray, Y: np.ndarray):
    """
    ## COST FUNCTION
    
    This method is to control the cost function to be returnes given the loss function picked and as well to return the gradient evaluation of the parameters\n
    
    ### Arguments:
    
    - X => input values of the examples
    - Y => output values of the examples
    
    ### Return:
    
    - J: {float} evaluated cost function given the chosen loss function type
    - grads_W: {np.ndarray} value containing all the gradient values of the W parameters of all the layers
    - grads_b: {np.ndarray} value containing all the gradient values of the b parameters of all the layers
    
    #### Notes:
    
    - This last two are return together as a tuple
    """
    A, _ = self.forward_propagation(X)
    if self.loss == 'mse': J = self.mse(X, Y)
    elif self.loss == 'bce': J = self.bce(X, Y)
    else: J = self.ce(X, Y)
    grads = self.backward_propagation(X, Y)
    grads_W = [g_ for i, g in enumerate(grads[0]) if i != 0 for g_ in g.flatten()]
    grads_b = [b_ for j, b in enumerate(grads[1]) if j != 0 for b_ in b.flatten()]
    return J, (np.array(grads_W), np.array(grads_b))

  def set_parameters(self, params_W: np.ndarray, params_b: np.ndarray):
    """
    ## SET PARAMETERS
    
    Set the parameters of the model from external sources\n
    
    ### Arguments:
    
    - params_W: {np.ndarray} with all the values respectively where they belong in all the W parameters
    - params_b: {np.ndarray} with all the values respectively where they belong in all the b parameters
    """
    end_W, end_b = 0, 0
    for i in range(len(self.layers_sizes)-1):
        start_W, start_b = end_W, end_b
        end_W += self.layers_sizes[i+1] * self.layers_sizes[i]
        end_b += self.layers_sizes[i+1]
        self.W[i+1] = np.asarray(params_W[start_W:end_W]).reshape(self.layers_sizes[i+1], self.layers_sizes[i])
        self.b[i+1] = np.asarray(params_b[start_b:end_b]).reshape(self.layers_sizes[i+1], 1)
  
  def get_parameters(self):
      """
      ## GET PARAMETERS
      
      Get the parameters of the model to use them in external sources
      
      ### Returns:
      
      - params_W => 1D array with all the values respectively where they belong in all the W parameters
      - params_b => 1D array with all the values respectively where they belong in all the b parameters
      
      * This variables are return together as a tuple
      """
      params_W = [w for i, W in enumerate(self.W) if i != 0 for w in W.flatten()]
      params_b = [b for j, B in enumerate(self.b) if j != 0 for b in B.flatten()]
      return np.array(params_W), np.array(params_b)
  
  def resume(self):
      """
      Prints a format with all the layers and amount of parameters in that given layer as well the final total amount of parameters.
      """
      pass