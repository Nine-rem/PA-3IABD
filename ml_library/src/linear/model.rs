// src/linear/model.rs

use pyo3::prelude::*;
use crate::common::math::dot_product;

/// Modèle linéaire
#[pyclass]
pub struct LinearModel {
    /// poids (biais en position 0, puis un poids par feature)
    #[pyo3(get, set)]
    pub weights: Vec<f64>,
}

// ——— Implémentation pure Rust (pour vos tests) ———
impl LinearModel {
    /// constructeur Rust pour les tests
    pub fn new_rust(n_features: usize) -> Self {
        LinearModel {
            weights: vec![0.0; n_features + 1],
        }
    }

    /// predict slice-based : prend un slice et renvoie un f64
    pub fn predict(&self, input: &[f64]) -> f64 {
        assert_eq!(
            input.len() + 1,
            self.weights.len(),
            "input.len()+1 must equal weights.len()"
        );
        self.weights[0] + dot_product(&self.weights[1..], input)
    }
}

// ——— Implémentation Python via PyO3 ———
#[pymethods]
impl LinearModel {
    /// nouveau constructeur Python : LinearModel(n_features)
    #[new]
    fn py_new(n_features: usize) -> Self {
        LinearModel::new_rust(n_features)
    }

    /// wrapper Python pour predict : prend Vec<f64> et renvoie PyResult<f64>
    #[pyo3(name = "predict")]
    fn predict_py(&self, input: Vec<f64>) -> PyResult<f64> {
        // on appelle la version slice-based
        Ok(self.predict(&input))
    }
}
