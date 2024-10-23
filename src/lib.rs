//! # Rapid-NN
//! 
//! **Rapid Neural Networks in Rust**: A lightweight and fast neural network library that's easy to use.
//! 
//! - **High Performance**: 4-5 times faster for simple networks on CPU than PyTorch.
//! - **Easy to Use**: Simple API for building and training neural networks.
//! 
//! ## Features
//! 
//! - Support for various activation functions.
//! - Customizable network architecture.
//! 
//! ## Example
//! 
//! ```rust
//! use rapid_nn::{ActivationFunction, NeuralNetwork};
//! 
//! fn main() {
//!   let mut nn = NeuralNetwork::new();
//! 
//!   // Define a network with an input layer, a hidden layer, and an output layer
//!   nn.add_layer(3, ActivationFunction::None, 0.0); // Input layer with 3 neurons
//!   nn.add_layer(5, ActivationFunction::LeakyReLU, 0.1); // Hidden layer with 5 neurons
//!   nn.add_layer(2, ActivationFunction::Linear, 0.1); // Output layer with 2 neurons
//! 
//!   // Define 10 predictable input-target pairs
//!   // Targets are linear combinations of inputs:
//!   // output1 = (x1 + x2 + x3) / 3
//!   // output2 = (x1 - x2 + x3) / 3
//!   let training_data = [
//!     ([0.1, 0.2, 0.3], [0.2, 0.0667]),
//!     ([0.4, 0.5, 0.6], [0.5, 0.1333]),
//!     ([0.2, 0.1, 0.4], [0.2333, 0.1667]),
//!     ([0.9, 0.8, 0.7], [0.8, 0.2667]),
//!     ([0.3, 0.4, 0.5], [0.4, 0.1333]),
//!     ([0.6, 0.5, 0.4], [0.5, 0.1667]),
//!     ([0.4, 0.3, 0.2], [0.3, 0.1]),
//!     ([0.7, 0.8, 0.9], [0.8, 0.2667]),
//!     ([0.2, 0.3, 0.1], [0.2, 0.0]),
//!     ([0.8, 0.7, 0.6], [0.7, 0.2333]),
//!   ];
//! 
//!   let epochs = 1000;
//! 
//!   for epoch in 1..=epochs {
//!     let mut total_error = 0.0;
//!     for (inputs, targets) in &training_data {
//!       let error = nn.train(inputs, targets, 0.01);
//!       total_error += error;
//!     }
//!     let avg_error = total_error / training_data.len() as f32;
//!     if epoch % 100 == 0 {
//!       println!("Epoch {}: Average Error = {:.6}", epoch, avg_error);
//!     }
//!   }
//! 
//!   // Make predictions on the training data
//!   for (inputs, targets) in &training_data {
//!     let output = nn.predict(inputs);
//!     println!(
//!       "Input: {:?} | Predicted Output: {:?} | Target: {:?}",
//!       inputs, output, targets
//!     );
//!   }
//! }
//! ```

use rand::random;

/// Activation functions.
#[derive(Clone, Copy)]
pub enum ActivationFunction {
  None,
  Identity,
  Linear,
  ReLU,
  LeakyReLU,
  ELU,
  Threshold,
  Sigmoid,
  Tanh,
}

/// Neural network.
pub struct NeuralNetwork {
  /// Number of layers in the network
  depth: usize,
  /// Number of neurons in each layer
  width: Vec<usize>,
  /// Weights between layers
  weight: Vec<Vec<Vec<f32>>>,
  /// Adjustments to weights during training
  weight_adj: Vec<Vec<Vec<f32>>>,
  /// Neuron activations
  neuron: Vec<Vec<f32>>,
  /// Loss gradients
  loss: Vec<Vec<f32>>,
  /// Pre-activation values
  preact: Vec<Vec<f32>>,
  /// Bias for each layer
  bias: Vec<f32>,
  /// Activation function for each layer
  activation: Vec<ActivationFunction>,
}

impl NeuralNetwork {
  pub fn new() -> Self {
    NeuralNetwork {
      depth: 0,
      width: Vec::new(),
      weight: Vec::new(),
      weight_adj: Vec::new(),
      neuron: Vec::new(),
      loss: Vec::new(),
      preact: Vec::new(),
      bias: Vec::new(),
      activation: Vec::new(),
    }
  }

  /// Adds a layer to the neural network.
  pub fn add_layer(&mut self, width: usize, activation: ActivationFunction, bias: f32) {
    self.depth += 1;
    self.width.push(width);
    self.activation.push(activation);
    self.bias.push(bias);

    if self.depth > 1 {
      // Initialize neurons, losses, and pre-activation values for the layer
      self.neuron.push(vec![0.0; width]);
      self.loss.push(vec![0.0; width]);
      self.preact.push(vec![0.0; width]);

      let prev_width = self.width[self.depth - 2];

      // Initialize weights and weight adjustments between layers
      let mut layer_weights = Vec::with_capacity(width);
      let mut layer_weight_adj = Vec::with_capacity(width);
      for _ in 0..width {
        let mut neuron_weights = Vec::with_capacity(prev_width);
        let mut neuron_weight_adj = Vec::with_capacity(prev_width);
        for _ in 0..prev_width {
          neuron_weights.push(random::<f32>() - 0.5); // Randomize weights
          neuron_weight_adj.push(0.0);
        }
        layer_weights.push(neuron_weights);
        layer_weight_adj.push(neuron_weight_adj);
      }
      self.weight.push(layer_weights);
      self.weight_adj.push(layer_weight_adj);
    } else {
      // Input layer placeholders
      self.neuron.push(Vec::new());
      self.loss.push(Vec::new());
      self.preact.push(Vec::new());
      self.weight.push(Vec::new());
      self.weight_adj.push(Vec::new());
    }
  }

  /// Performs forward propagation through the network.
  fn forward_propagation(&mut self) {
    for i in 1..self.depth {
      for j in 0..self.width[i] {
        let mut sum = 0.0;
        for k in 0..self.width[i - 1] {
          sum += self.neuron[i - 1][k] * self.weight[i][j][k];
        }
        sum += self.bias[i];
        self.preact[i][j] = sum;
        self.neuron[i][j] = activation_function(sum, false, self.activation[i]);
      }
    }
  }

  /// Trains the neural network with the given inputs and targets.
  pub fn train(&mut self, inputs: &[f32], targets: &[f32], rate: f32) -> f32 {
    self.neuron[0] = inputs.to_vec();
    self.forward_propagation();

    let mut err = 0.0;
    let output_layer = self.depth - 1;

    // Calculate loss at the output layer
    for j in 0..self.width[output_layer] {
      self.loss[output_layer][j] =
        error_derivative(targets[j], self.neuron[output_layer][j]);
      err += error(targets[j], self.neuron[output_layer][j]);
    }

    // Backpropagate the loss
    for i in (1..output_layer).rev() {
      for j in 0..self.width[i] {
        let mut sum = 0.0;
        for k in 0..self.width[i + 1] {
          sum += self.loss[i + 1][k]
            * activation_function(
            self.preact[i + 1][k],
            true,
            self.activation[i + 1],
          )
            * self.weight[i + 1][k][j];
        }
        self.loss[i][j] = sum;
      }
    }

    // Calculate weight adjustments
    for i in (1..self.depth).rev() {
      for j in 0..self.width[i] {
        for k in 0..self.width[i - 1] {
          self.weight_adj[i][j][k] = self.loss[i][j]
            * activation_function(self.preact[i][j], true, self.activation[i])
            * self.neuron[i - 1][k];
        }
      }
    }

    // Apply weight adjustments
    for i in (1..self.depth).rev() {
      for j in 0..self.width[i] {
        for k in 0..self.width[i - 1] {
          self.weight[i][j][k] += self.weight_adj[i][j][k] * rate;
        }
      }
    }

    err
  }

  /// Predicts the output for the given inputs.
  pub fn predict(&mut self, inputs: &[f32]) -> &[f32] {
    self.neuron[0] = inputs.to_vec();
    self.forward_propagation();
    &self.neuron[self.depth - 1]
  }
}

/// Computes the error between target and output.
fn error(a: f32, b: f32) -> f32 {
  0.5 * (a - b) * (a - b)
}

/// Computes the derivative of the error.
fn error_derivative(a: f32, b: f32) -> f32 {
  a - b
}

/// The main activation function dispatcher.
fn activation_function(a: f32, derivative: bool, function_type: ActivationFunction) -> f32 {
  match function_type {
    ActivationFunction::None => activation_function_none(a, derivative),
    ActivationFunction::Identity => activation_function_identity(a, derivative),
    ActivationFunction::Linear => activation_function_linear(a, derivative),
    ActivationFunction::ReLU => activation_function_relu(a, derivative),
    ActivationFunction::LeakyReLU => activation_function_leaky_relu(a, derivative),
    ActivationFunction::ELU => activation_function_elu(a, derivative),
    ActivationFunction::Threshold => activation_function_threshold(a, derivative),
    ActivationFunction::Sigmoid => activation_function_sigmoid(a, derivative),
    ActivationFunction::Tanh => activation_function_tanh(a, derivative),
  }
}

/// Null activation function.
fn activation_function_none(_a: f32, _derivative: bool) -> f32 {
  0.0
}

/// Identity activation function.
fn activation_function_identity(a: f32, derivative: bool) -> f32 {
  if derivative {
    1.0
  } else {
    a
  }
}

/// Linear activation function.
fn activation_function_linear(a: f32, derivative: bool) -> f32 {
  if derivative {
    1.0
  } else {
    a
  }
}

/// ReLU activation function.
fn activation_function_relu(a: f32, derivative: bool) -> f32 {
  if a >= 0.0 {
    if derivative {
      1.0
    } else {
      a
    }
  } else {
    0.0
  }
}

/// Leaky ReLU activation function.
fn activation_function_leaky_relu(a: f32, derivative: bool) -> f32 {
  if a > 0.0 {
    if derivative {
      1.0
    } else {
      a
    }
  } else {
    if derivative {
      0.01
    } else {
      a * 0.01
    }
  }
}

/// ELU activation function.
fn activation_function_elu(a: f32, derivative: bool) -> f32 {
  if a >= 0.0 {
    if derivative {
      1.0
    } else {
      a
    }
  } else {
    if derivative {
      activation_function_elu(a, false)
    } else {
      a.exp() - 1.0
    }
  }
}

/// Threshold activation function.
fn activation_function_threshold(a: f32, derivative: bool) -> f32 {
  if derivative {
    0.0
  } else if a > 0.0 {
    1.0
  } else {
    0.0
  }
}

/// Sigmoid activation function.
fn activation_function_sigmoid(a: f32, derivative: bool) -> f32 {
  if derivative {
    let f = activation_function_sigmoid(a, false);
    f * (1.0 - f)
  } else {
    1.0 / (1.0 + (-a).exp())
  }
}

/// Tanh activation function.
fn activation_function_tanh(a: f32, derivative: bool) -> f32 {
  if derivative {
    1.0 - activation_function_tanh(a, false).powi(2)
  } else {
    (2.0 / (1.0 + (-2.0 * a).exp())) - 1.0
  }
}
