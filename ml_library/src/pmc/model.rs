// src/pmc/model.rs

/// Perceptron multiclasses (one-vs-rest)
/// weights[class][j]: biais en j=0, puis poids j-1
#[derive(Debug)]
pub struct PMC {
    pub weights: Vec<Vec<f64>>,
}

impl PMC {
    /// Prédit un vecteur de scores (un par classe) pour l'entrée `x`
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            w[0] + w[1..]
                .iter()
                .zip(x)
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>()
        }).collect()
    }
}