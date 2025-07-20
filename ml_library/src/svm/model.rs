use crate::common::math::dot_product;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SVM {
    pub weights: Vec<f64>,  // w0 = biais (intercept)
    pub learning_rate: f64,
    pub lambda: f64,        // régularisation
}

#[pymethods]
impl SVM {
    #[new]
    pub fn new_py(n_features: usize, learning_rate: f64, lambda: f64) -> Self {
        Self {
            weights: vec![0.0; n_features + 1],
            learning_rate,
            lambda,
        }
    }

    /// Calcule la sortie du SVM (score non borné)
    pub fn decision_function(&self, input: Vec<f64>) -> f64 {
        assert_eq!(
            input.len() + 1,
            self.weights.len(),
            "input.len()+1 doit == weights.len()"
        );
        // w · x + b
        let dot = dot_product(&self.weights[1..], &input);
        dot + self.weights[0]
    }

    /// Prédit la classe (+1 ou -1)
    pub fn predict(&self, input: Vec<f64>) -> f64 {
        if self.decision_function(input) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
