use crate::common::math::{dot_product, sub_vectors};
use nalgebra::{DMatrix, DVector};

/// Réseau RBF simple
pub struct RBFN {
    pub centers: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub gamma: f64,
}

impl RBFN {
    /// Crée un RBFN non entraîné
    pub fn new(centers: Vec<Vec<f64>>, gamma: f64) -> Self {
        assert!(!centers.is_empty(), "Il faut au moins un centre");
        let weights = vec![0.0; centers.len()];
        Self { centers, weights, gamma }
    }

    /// Fonction RBF gaussienne
    pub fn rbf(&self, x: &[f64], center: &[f64]) -> f64 {
        assert_eq!(x.len(), center.len(), "Dim(x) doit == Dim(center)");
        let diff = sub_vectors(x, center);
        let dist2 = dot_product(&diff, &diff);
        (-self.gamma * dist2).exp()
    }

    /// Calcule la sortie pour une entrée donnée
    pub fn predict(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.centers[0].len(), "Dim(x) incohérente");
        self.centers.iter()
            .zip(&self.weights)
            .map(|(c, w)| w * self.rbf(x, c))
            .sum()
    }

    /// Calcule Φ (matrice des activations)
    pub fn compute_phi(&self, inputs: &[Vec<f64>]) -> DMatrix<f64> {
        let n_samples = inputs.len();
        let n_centers = self.centers.len();
        let mut phi = DMatrix::zeros(n_samples, n_centers);

        for i in 0..n_samples {
            for j in 0..n_centers {
                phi[(i, j)] = self.rbf(&inputs[i], &self.centers[j]);
            }
        }
        phi
    }

    /// Applique les poids calculés (W = (Φᵀ Φ)⁻¹ Φᵀ Y)
    pub fn fit_weights(&mut self, phi: &DMatrix<f64>, y: &DVector<f64>) {
        // Optionnel : vérifier dims
        assert_eq!(phi.nrows(), y.len(), "φ.rows doit == y.len()");
        let gram = phi.transpose() * phi;
        let inv = gram.clone()
            .try_inverse()
            .unwrap_or_else(|| gram.pseudo_inverse(1e-8).unwrap());
        let w = inv * phi.transpose() * y;
        self.weights = w.iter().copied().collect();
    }
}
