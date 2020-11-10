import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, noinputs, loss):
        """
        Artificial Neural Network\n
        Arguments:\n
        \tnoinputs => Number inputs that the model will be receiving, the inputs should have a shape (n, m) where n is the value to set and m is the number of examples
        \tloss => The type of loss function to be used, can be:
        \t\t- Mean Squared Error (pass in: 'mse')
        \t\t- Binary Cross Entropy (pass in: 'bce')
        \t\t- Cross Entropy (pass in: 'ce')
        """
        self.layers_sizes = [noinputs]
        self.loss = loss
        self.W, self.b, self.act_funcs, self.keep_probs = [0], [0], ['linear'], [1]
    
    def xavier_initialization(self, n0, n1):
        """
        XAVIER INITIALIZATION\n
        Random initialization for parameters of deep neural network scaled depending on the size of the layers that surround the parameters\n
        Arguments:\n
        \tn0 => shape of the layer that comes before parameters, also the column size of the output array
        \tn1 => shape of the layer that comes after parameters, also the row size of the output array
        Return:\n
        \tArray with shape (n1, n0)
        """
        return np.random.randn(n1, n0) * ((2 / (n0 + n1))**0.5)

    def push_layer(self, n_units=None, activation_function=None, keep_prob=1):
        """
        LAYER ADDER\n
        Arguments:\n
        \tn-units => number of units to be added in the current layer
        \tactivation-function => desired activation function for the current layer, can be:
        \t\t- Sigmoid (pass in: 'sigmoid')
        \t\t- Hyperbolic Tangent (pass in: 'tanh')
        \t\t- REctified Linear Unit (pass in: 'relu')
        \t\t- LEAKY REctified Linear Unit (pass in: 'leaky_relu')
        \tkeep-prob => probability to keep different units of the layer, as form to apply dropout layer\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        self.layers_sizes.append(n_units)
        self.W.append(self.xavier_initialization(self.layers_sizes[-2], self.layers_sizes[-1]))
        self.b.append(np.zeros((self.layers_sizes[-1], 1)))
        self.act_funcs.append(activation_function)
        self.keep_probs.append(keep_prob)

    def sigmoid(self, z, is_grad=False):
        """
        SIGMOID ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the sigmoid activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the sigmoid function
        Return:\n
        \tEvaluated sigmoid function, it depends if is the derivative evaluation on the is-grad variable\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        if is_grad:
            s = self.sigmoid(z, False)
            return s * (1 - s)
        else:
            return 1 / (1 + np.exp(-z))
    
    def tanh(self, z, is_grad=False):
        """
        HYPERBOLIC TANGENT ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the tanh activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the tanh function
        Return:\n
        \tEvaluated tanh function, it depends if is the derivative evaluation on the is-grad variable\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        if is_grad:
            t = self.tanh(z, False)
            return 1 - (t ** 2)
        else:
            return np.tanh(z)
    
    def relu(self, z, is_grad=False):
        """
        RECTIFIED LINEAR UNIT ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the relu activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the relu function
        Return:\n
        \tEvaluated relu function, it depends if is the derivative evaluation on the is-grad variable\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        return np.where(z < 0, 0, 1) if is_grad else np.maximum(0, z)
    
    def leaky_relu(self, z, is_grad=False):
        """
        LEAKY RECTIFIED LINEAR UNIT ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the leaky relu activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the leaky relu function
        Return:\n
        \tEvaluated leaky relu function, it depends if is the derivative evaluation on the is-grad variable\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        return np.where(z < 0, 0.01, 1) if is_grad else np.maximum(0.01 * z, z)
    
    def linear(self, z, is_grad=False):
        """
        LINEAR UNIT ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the linear activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the linear function
        Return:\n
        \tEvaluated linear function, it depends if is the derivative evaluation on the is-grad variable\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        return 1 if is_grad else z
    
    def softmax(self, z):
        """
        SOFTMAX ACTIVATION FUNCTION\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the softmax activation function
        Return:\n
        \tEvaluated leaky relu function
        """
        t = np.exp(z)
        return t / np.sum(t)
    
    def activation_function(self, z, is_grad, act_func):
        """
        GENERAL ACTIVATION FUNCTION GETTER FOR THE OBJECT\n
        The objective of this method is to control all the available activation functions\n
        Arguments:\n
        \tz => ndarray or number which is needed to be evaluated through the softmax activation function
        \tis-grad => boolean value which defines if the value to returned is the derivative or not of the evaluation of the linear function
        \tact-func => string dictating the desired activation function, can be:
        \t\t-'sigmoid'
        \t\t-'tanh'
        \t\t-'relu'
        \t\t-'leaky-relu'
        \t\t-'linear'
        \t\t-'softmax'
        Return:\n
        \tEvaluated respective activation function given the variable act-func\n
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        if act_func == 'sigmoid': return self.sigmoid(z, is_grad)
        elif act_func == 'tanh': return self.tanh(z, is_grad)
        elif act_func == 'relu': return self.relu(z, is_grad)
        elif act_func == 'leaky_relu': return self.leaky_relu(z, is_grad)
        elif act_func == 'linear': return self.linear(z, is_grad)
        elif act_func == 'softmax': return self.softmax(z)
        else: return 'Error'

    def mse(self, X, Y):
        """
        MEAN SQUARED ERROR LOSS FUNCTION\n
        Arguments:\n
        \tX => input values of the examples
        \tY => output values of the examples
        Return:\n
        \tEvaluated mean squared error loss function in all the examples between the predicted value given by the variable X and the variable Y
        """
        A, _ = self.forward_propagation(X)
        dif_Y = A[-1] - Y
        return (dif_Y @ dif_Y.T).squeeze().squeeze()/X.shape[1]
    
    def bce(self, X, Y):
        """
        BINARY CROSS ENTROPY LOSS FUNCTION\n
        Arguments:\n
        \tX => input values of the examples
        \tY => output values of the examples
        Return:\n
        \tEvaluated log-like loss function in all the examples for binary class discrete models between the predicted value given by the variable X and the variable Y
        """
        A, _ = self.forward_propagation(X)
        return -(Y @ np.log(A[-1]).T + (1 - Y) @ np.log(1 - A[-1]).T).squeeze().squeeze()/X.shape[1]
    
    def ce(self, X, Y):
        """
        CROSS ENTROPY LOSS FUNCTION\n
        Arguments:\n
        \tX => input values of the examples
        \tY => output values of the examples
        Return:\n
        \tEvaluated log-like loss function in all the examples between for multi class discrete models the predicted value given by the variable X and the variable Y
        """
        A, _ = self.forward_propagation(X)
        return -np.sum(np.log(A[-1]) @ Y.T)/X.shape[1]
    
    def forward_propagation(self, X):
        """
        FORWARD PROPAGATION LOSS FUNCTION\n
        Arguments:\n
        \tX => input values of the examples
        Return:\n
        \tAll the evaluated units activated and not activated values, stored in the variables A and Z, respectively.
        """
        A, Z = [X], [None]
        for i in range(1, len(self.layers_sizes)):
            Z.append(self.W[i] @ A[i-1] + self.b[i])
            a = self.activation_function(Z[i], False, self.act_funcs[i])
            d = np.random.rand(a.shape[0], a.shape[1]) < self.keep_probs[i]
            A.append((a * d) / self.keep_probs[i])
        return A, Z

    def backward_propagation(self, X, Y):
        """
        BACKWARD PROPAGATION LOSS FUNCTION\n
        Arguments:\n
        \tX => input values of the examples
        \tY => output values of the examples
        Return:\n
        \tdW => Gradient of the W paramaters
        \tdb => Gradient of the b parameters
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
    
    def cost_function(self, X, Y):
        """
        COST FUNCTION\n
        \nThis method is to control the cost function to be returnes given the loss function picked and as well to return the gradient evaluation of the parameters\n
        Arguments:\n
        \tX => input values of the examples
        \tY => output values of the examples
        Return:\n
        \tJ => evaluated cost function given the chosen loss function type
        \tgrads-W => Array like variable containing all the gradient values of the W parameters of all the layers
        \tgrads-b => Array like variable containing all the gradient values of the b parameters of all the layers\n
        \t* Note: this last two are return together as a tuple
        * Note: variable names do not use - but underscore, the change is because underscore means something in pydocs jeje
        """
        if self.loss == 'mse': J = self.mse(X, Y)
        elif self.loss == 'bce': J = self.bce(X, Y)
        elif self.loss == 'ce': J = self.ce(X, Y)
        grads = self.backward_propagation(X, Y)
        grads_W = [g_ for i, g in enumerate(grads[0]) if i != 0 for g_ in g.flatten()]
        grads_b = [b_ for j, b in enumerate(grads[1]) if j != 0 for b_ in b.flatten()]
        return J, (np.array(grads_W), np.array(grads_b))

    def set_parameters(self, params_W, params_b):
        """
        SET PARAMETERS\n
        Set the parameters of the model from external sources\n
        Arguments:\n
        \tparams-W => 1D array with all the values respectively where they belong in all the W parameters
        \tparams-b => 1D array with all the values respectively where they belong in all the b parameters
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
        GET PARAMETERS\n
        Get the parameters of the model to use them in external sources\n
        Returns:\n
        \tparams-W => 1D array with all the values respectively where they belong in all the W parameters
        \tparams-b => 1D array with all the values respectively where they belong in all the b parameters
        \t* This variables are return together as a tuple
        """
        params_W = [w for i, W in enumerate(self.W) if i != 0 for w in W.flatten()]
        params_b = [b for j, B in enumerate(self.b) if j != 0 for b in B.flatten()]
        return np.array(params_W), np.array(params_b)
    
    def resume(self):
        """
        Prints a format with all the layers and amount of parameters in that given layer as well the final total amount of parameters.
        """
        pass