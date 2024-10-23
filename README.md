# Rapid-NN

**Rapid Neural Networks in Rust**: A lightweight and fast neural network library that's easy to use.

- **High Performance**: 4-5 times faster for simple networks on CPU than PyTorch.
- **Easy to Use**: Simple API for building and training neural networks.

## Features

- Support for various activation functions.
- Customizable network architecture.

## Example

```rust
use rapid_nn::{ActivationFunction, NeuralNetwork};

fn main() {
  let mut nn = NeuralNetwork::new();

  // Define a network with an input layer, a hidden layer, and an output layer
  nn.add_layer(3, ActivationFunction::None, 0.0); // Input layer with 3 neurons
  nn.add_layer(5, ActivationFunction::LeakyReLU, 0.1); // Hidden layer with 5 neurons
  nn.add_layer(2, ActivationFunction::Linear, 0.1); // Output layer with 2 neurons

  // Define 10 predictable input-target pairs
  // Targets are linear combinations of inputs:
  // output1 = (x1 + x2 + x3) / 3
  // output2 = (x1 - x2 + x3) / 3
  let training_data = [
    ([0.1, 0.2, 0.3], [0.2, 0.0667]),
    ([0.4, 0.5, 0.6], [0.5, 0.1333]),
    ([0.2, 0.1, 0.4], [0.2333, 0.1667]),
    ([0.9, 0.8, 0.7], [0.8, 0.2667]),
    ([0.3, 0.4, 0.5], [0.4, 0.1333]),
    ([0.6, 0.5, 0.4], [0.5, 0.1667]),
    ([0.4, 0.3, 0.2], [0.3, 0.1]),
    ([0.7, 0.8, 0.9], [0.8, 0.2667]),
    ([0.2, 0.3, 0.1], [0.2, 0.0]),
    ([0.8, 0.7, 0.6], [0.7, 0.2333]),
  ];

  let epochs = 1000;

  for epoch in 1..=epochs {
    let mut total_error = 0.0;
    for (inputs, targets) in &training_data {
      let error = nn.train(inputs, targets, 0.01);
      total_error += error;
    }
    let avg_error = total_error / training_data.len() as f32;
    if epoch % 100 == 0 {
      println!("Epoch {}: Average Error = {:.6}", epoch, avg_error);
    }
  }

  // Make predictions on the training data
  for (inputs, targets) in &training_data {
    let output = nn.predict(inputs);
    println!(
      "Input: {:?} | Predicted Output: {:?} | Target: {:?}",
      inputs, output, targets
    );
  }
}
```
