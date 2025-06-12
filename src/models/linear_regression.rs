use crate::models::model_trait::Model;
use crate::models::utils::{transpose, matmul, matvecmul, inverse};

pub struct LinearRegression {
    pub weights: Vec<f64>,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { weights: Vec::new() }
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
}

impl Model for LinearRegression {
    type Input = Vec<f64>;
    type Output = f64;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        let mut mat_x = vec![vec![1.0; n_features + 1]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_features {
                mat_x[i][j + 1] = x[i][j];
            }
        }

        let xt = transpose(&mat_x);
        let xtx = matmul(&xt, &mat_x);
        let xtx_inv = inverse(&xtx).expect("Matrice X^T X non inversible.");
        let xty = matvecmul(&xt, y);

        self.weights = matvecmul(&xtx_inv, &xty);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        if self.weights.is_empty() {
            panic!("Modèle non entraîné");
        }
        let mut sum = self.weights[0];
        for i in 0..x.len() {
            sum += self.weights[i + 1] * x[i];
        }
        sum
    }
}
