use crate::models::model_trait::Model;

/// Support Vector Machine (SVM) linéaire avec descente de gradient et régularisation L2.
pub struct SVM {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
}

impl SVM {
    /// Initialise un SVM avec dimension d’entrée, learning rate, epochs et régularisation lambda.
    pub fn new(input_dim: usize, learning_rate: f64, epochs: usize, lambda: f64) -> Self {
        SVM {
            weights: vec![0.0; input_dim],
            bias: 0.0,
            learning_rate,
            epochs,
            lambda,
        }
    }

    /// Getter poids
    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    /// Setter poids
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    /// Getter biais
    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    /// Setter biais
    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    /// Prédiction interne : produit scalaire + biais.
    fn predict_raw(&self, x: &Vec<f64>) -> f64 {
        self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + self.bias
    }
}

impl Model for SVM {
    type Input = Vec<f64>;
    type Output = i8;

    /// Entraînement par descente de gradient avec hinge loss.
    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        for _epoch in 0..self.epochs {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let cond = yi as f64 * self.predict_raw(xi);
                if cond >= 1.0 {
                    // Pas de mise à jour de biais, poids pénalisés par régularisation
                    for i in 0..self.weights.len() {
                        self.weights[i] -= self.learning_rate * 2.0 * self.lambda * self.weights[i];
                    }
                } else {
                    // Mise à jour poids + biais
                    for i in 0..self.weights.len() {
                        self.weights[i] += self.learning_rate * (yi as f64 * xi[i] - 2.0 * self.lambda * self.weights[i]);
                    }
                    self.bias += self.learning_rate * yi as f64;
                }
            }
        }
    }

    /// Prédiction finale : signe de la sortie linéaire.
    fn predict(&self, x: &Self::Input) -> Self::Output {
        if self.predict_raw(x) >= 0.0 { 1 } else { -1 }
    }
}
