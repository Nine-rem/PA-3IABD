use crate::models::model_trait::Model;

/// Modèle linéaire simple de classification binaire (Perceptron).
pub struct LinearClassification {
    pub weights: Vec<f64>, // poids avec biais en position 0
}

impl LinearClassification {
    /// Initialise un modèle vide sans poids.
    pub fn new() -> Self {
        LinearClassification { weights: Vec::new() }
    }

    /// Setter poids (utile pour tests).
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    /// Getter poids.
    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
}

impl Model for LinearClassification {
    type Input = Vec<f64>;
    type Output = i8;

    /// Entraînement avec règle de Rosenblatt (Perceptron simple).
    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Initialisation poids (biais + poids) à zéro
        self.weights = vec![0.0; n_features + 1];

        let learning_rate = 0.1;
        let max_iter = 1000;

        for _ in 0..max_iter {
            let mut errors = 0;
            for i in 0..n_samples {
                let prediction = self.predict(&x[i]);
                let error = y[i] - prediction;
                if error != 0 {
                    errors += 1;
                    self.weights[0] += learning_rate * error as f64; // biais
                    for j in 0..n_features {
                        self.weights[j + 1] += learning_rate * error as f64 * x[i][j];
                    }
                }
            }
            if errors == 0 {
                break; // convergence atteinte
            }
        }
    }

    /// Prédiction : signe du produit scalaire + biais.
    fn predict(&self, x: &Self::Input) -> Self::Output {
        if self.weights.is_empty() {
            panic!("Modèle non entraîné");
        }
        let mut sum = self.weights[0];
        for i in 0..x.len() {
            sum += self.weights[i + 1] * x[i];
        }
        if sum >= 0.0 { 1 } else { -1 }
    }
}
