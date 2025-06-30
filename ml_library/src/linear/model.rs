//régression continue
pub struct LinearModel {
    pub weights: Vec<f64>,
}

impl LinearModel {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features + 1], // biais + poids
        }
    }

    pub fn predict(&self, input: &[f64]) -> f64 {
        self.weights[0] + self.weights[1..]
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
    }
}
