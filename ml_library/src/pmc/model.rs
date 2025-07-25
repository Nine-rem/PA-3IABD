
use ndarray::{Array1, Array2};
use rand::distributions::{Uniform, Distribution};
use rand::thread_rng;

/// Structure PMC simple
pub struct PMC {
    pub layers: Vec<usize>,
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
    pub learning_rate: f32,
    pub epochs: usize,
    pub is_classification: bool,
}

impl PMC {
    /// Constructeur simple
    pub fn new(
        layers: Vec<usize>,
        learning_rate: f32,
        epochs: usize,
        is_classification: bool,
    ) -> Self {
        let mut rng = thread_rng();
        let dist = Uniform::new(-0.5f32, 0.5f32);
        let mut weights = Vec::with_capacity(layers.len() - 1);
        let mut biases = Vec::with_capacity(layers.len() - 1);
        
        for i in 0..layers.len() - 1 {
            let w = Array2::from_shape_fn((layers[i], layers[i + 1]), |_| dist.sample(&mut rng));
            weights.push(w);
            biases.push(Array1::zeros(layers[i + 1]));
        }
        
        PMC { 
            layers, 
            weights, 
            biases, 
            learning_rate, 
            epochs, 
            is_classification,
        }
    }

    /// Extraire tous les poids sous forme de vecteur aplati
    pub fn get_flattened_weights(&self) -> Vec<f32> {
        let mut flattened = Vec::new();
        for (i, weight_matrix) in self.weights.iter().enumerate() {
            // Ajouter les poids
            for row in weight_matrix.outer_iter() {
                flattened.extend(row.iter());
            }
            // Ajouter les biais
            flattened.extend(self.biases[i].iter());
        }
        flattened
    }

    // Fonctions d'activation simples
    pub fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v.clamp(-500.0, 500.0)).exp()))
    }
    
    pub fn sigmoid_derivative(x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v * (1.0 - v))
    }
    
    pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = x.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        exp / sum
    }
}