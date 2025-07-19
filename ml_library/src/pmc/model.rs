use rand::Rng;
use crate::common::activations::tanh;

/// Représente une couche du MLP
#[derive(Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>, // [n_inputs+1][n_neurons]
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

    /// Forward pass sur la couche : tanh(w·x + b)
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(inputs.len() + 1, self.weights.len(),
            "inputs.len()+1 must equal weights.len()");
        let bias_row = &self.weights[inputs.len()];
        let mut out = vec![0.0; self.outputs.len()];
        for j in 0..out.len() {
            // biais
            let mut sum = bias_row[j];
            // somme pondérée
            for (i, &x) in inputs.iter().enumerate() {
                sum += x * self.weights[i][j];
            }
            out[j] = tanh(sum);
        }
        self.outputs = out.clone();
        out
    }
}

/// MLP complet
#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

impl MLP {
    pub fn new(
        n_inputs: usize,
        hidden_sizes: &[usize],
        n_outputs: usize,
        learning_rate: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev = n_inputs;
        for &h in hidden_sizes {
            layers.push(Layer::new(prev, h));
            prev = h;
        }
        layers.push(Layer::new(prev, n_outputs));
        Self { layers, learning_rate }
    }

    /// Forward complet (modifie les .outputs de chaque couche)
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut activation = input.to_vec();
        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }
        activation
    }

    /// Alias sans effet de bord supplémentaire
    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.forward(input)
    }
}
