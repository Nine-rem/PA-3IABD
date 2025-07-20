use pyo3::prelude::*;
use rand::Rng;
use crate::common::activations::tanh;

#[derive(Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub outputs: Vec<f64>,
    pub deltas: Vec<f64>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![vec![0.0; n_neurons]; n_inputs + 1];
        for i in 0..=n_inputs {
            for j in 0..n_neurons {
                weights[i][j] = rng.gen_range(-1.0..1.0);
            }
        }
        Self {
            weights,
            outputs: vec![0.0; n_neurons],
            deltas: vec![0.0; n_neurons],
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let bias_row = &self.weights[inputs.len()];
        let mut out = vec![0.0; self.outputs.len()];
        for j in 0..out.len() {
            let mut sum = bias_row[j];
            for (i, &x) in inputs.iter().enumerate() {
                sum += x * self.weights[i][j];
            }
            out[j] = tanh(sum);
        }
        self.outputs = out.clone();
        out
    }
}

#[pyclass]
pub struct MLP {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

#[pymethods]
impl MLP {
    #[new]
    pub fn new_py(n_inputs: usize, hidden_sizes: Vec<usize>, n_outputs: usize, learning_rate: f64) -> Self {
        let mut layers = Vec::new();
        let mut prev = n_inputs;
        for &h in &hidden_sizes {
            layers.push(Layer::new(prev, h));
            prev = h;
        }
        layers.push(Layer::new(prev, n_outputs));
        Self { layers, learning_rate }
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut activation = input;
        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }
        activation
    }
}

// Rust-only impl
impl MLP {
    pub fn predict_internal(&mut self, input: &[f64]) -> Vec<f64> {
        let mut activation = input.to_vec();
        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }
        activation
    }
}
