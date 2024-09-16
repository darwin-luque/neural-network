# Neural Network from Scratch

This project implements a basic neural network from scratch using Python. The network can be trained to classify or predict based on input data without the use of external machine learning libraries.

## Features

- Forward propagation
- Backpropagation
- Adjustable learning rate
- Multi-layer architecture

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/darwin-luque/neural-network.git
cd neural-network
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python testing.py
```

> **Note**: Generate your own dataset. Maybe try using the XOR problem to test the network.

## How It Works

- Forward Propagation: The network computes the output by passing inputs through multiple layers.
- Backpropagation: The error is computed and propagated back to update weights.

### Example

To showcase how the neural network works, you can run it on a simple dataset (e.g., XOR problem) and evaluate the model’s accuracy.

## Future Improvements

- Adding more complex activation functions like ReLU or Tanh.
- Implementing optimization techniques like momentum or learning rate decay.
- Expanding the network to include convolutional neural networks (CNNs) or recurrent neural networks (RNNs).
