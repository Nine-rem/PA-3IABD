use crate::common::math::dot_product;

pub struct LinearModel {
    pub weights: Vec<f64>,
}

impl LinearModel {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features + 1],
        }
    }

    pub fn predict(&self, input: &[f64]) -> f64 {
        assert_eq!(
            input.len() + 1,
            self.weights.len(),
            "input.len()+1 must equal weights.len()"
        );
        self.weights[0] + dot_product(&self.weights[1..], input)
    }
}
