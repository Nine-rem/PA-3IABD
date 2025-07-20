use pyo3::prelude::*;
use rand::Rng;

#[pyclass]
pub struct Perceptron {
    #[pyo3(get, set)]
    pub weights: Vec<f64>,
    #[pyo3(get, set)]
    pub learning_rate: f64,
}

#[pymethods]
impl Perceptron {
    #[new]
    pub fn new_py(n_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..=n_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Self { weights, learning_rate }
    }

    pub fn predict(&self, input: Vec<f64>) -> f64 {
        assert_eq!(input.len() + 1, self.weights.len(), "input.len()+1 doit == weights.len()");
        let sum = self.weights[0] + self.weights[1..]
            .iter()
            .zip(&input)
            .map(|(w, x)| w * x)
            .sum::<f64>();
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }
}

// Version Rust-only
impl Perceptron {
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..=n_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Self { weights, learning_rate }
    }

    pub fn predict_internal(&self, input: &[f64]) -> f64 {
        assert_eq!(input.len() + 1, self.weights.len(), "input.len()+1 doit == weights.len()");
        let sum = self.weights[0] + self.weights[1..]
            .iter()
            .zip(input)
            .map(|(w, x)| w * x)
            .sum::<f64>();
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }
}
