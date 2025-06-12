use crate::models::model_trait::Model;

/// Structure représentant un Perceptron Multi-Couches (MLP) simple avec une couche cachée.
pub struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w_input_hidden: Vec<Vec<f64>>,  // poids de la couche d'entrée vers couche cachée
    w_hidden_output: Vec<Vec<f64>>, // poids de la couche cachée vers sortie
    b_hidden: Vec<f64>,              // biais de la couche cachée
    b_output: Vec<f64>,              // biais de la couche de sortie
    learning_rate: f64,
}

impl MLP {
    /// Initialise un MLP avec tailles des couches et taux d'apprentissage.
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let input_size = layer_sizes[0];
        let hidden_size = layer_sizes[1];
        let output_size = layer_sizes[2];

        let mut w_input_hidden = vec![vec![0.0; hidden_size]; input_size];
        let mut w_hidden_output = vec![vec![0.0; output_size]; hidden_size];

        let mut rng = rand::thread_rng();
        for i in 0..input_size {
            for j in 0..hidden_size {
                w_input_hidden[i][j] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            }
        }
        for j in 0..hidden_size {
            for k in 0..output_size {
                w_hidden_output[j][k] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            }
        }

        MLP {
            input_size,
            hidden_size,
            output_size,
            w_input_hidden,
            w_hidden_output,
            b_hidden: vec![0.0; hidden_size],
            b_output: vec![0.0; output_size],
            learning_rate,
        }
    }

    /// Entraîne le MLP avec rétropropagation.
    pub fn train(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<Vec<f64>>, epochs: usize) {
        for _ in 0..epochs {
            for (x, y) in x_train.iter().zip(y_train.iter()) {
                let (hidden_out, outputs) = self.forward(x);
                let output_errors: Vec<f64> = outputs.iter().zip(y.iter())
                    .map(|(o, y_true)| y_true - o)
                    .collect();
                let hidden_errors = self.backward(&output_errors, &hidden_out);
                self.update_weights(x, &hidden_out, &output_errors, &hidden_errors);
            }
        }
    }

    /// Passe avant (forward pass) avec activation tanh sur couche cachée.
    pub fn forward(&self, x: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut hidden_out = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut sum = self.b_hidden[j];
            for i in 0..self.input_size {
                sum += x[i] * self.w_input_hidden[i][j];
            }
            hidden_out[j] = sum.tanh();
        }
        let mut outputs = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            let mut sum = self.b_output[k];
            for j in 0..self.hidden_size {
                sum += hidden_out[j] * self.w_hidden_output[j][k];
            }
            outputs[k] = sum;
        }
        (hidden_out, outputs)
    }

    /// Calcul des erreurs cachées (rétropropagation).
    fn backward(&self, output_errors: &Vec<f64>, hidden_out: &Vec<f64>) -> Vec<f64> {
        let mut hidden_errors = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut error = 0.0;
            for k in 0..self.output_size {
                error += output_errors[k] * self.w_hidden_output[j][k];
            }
            hidden_errors[j] = error * (1.0 - hidden_out[j].powi(2)); // dérivée tanh
        }
        hidden_errors
    }

    /// Mise à jour des poids et biais.
    fn update_weights(&mut self, x: &Vec<f64>, hidden_out: &Vec<f64>, output_errors: &Vec<f64>, hidden_errors: &Vec<f64>) {
        for k in 0..self.output_size {
            self.b_output[k] += self.learning_rate * output_errors[k];
            for j in 0..self.hidden_size {
                self.w_hidden_output[j][k] += self.learning_rate * output_errors[k] * hidden_out[j];
            }
        }
        for j in 0..self.hidden_size {
            self.b_hidden[j] += self.learning_rate * hidden_errors[j];
            for i in 0..self.input_size {
                self.w_input_hidden[i][j] += self.learning_rate * hidden_errors[j] * x[i];
            }
        }
    }

    /// Prédit la sortie pour un vecteur d’entrée.
    pub fn predict(&self, x: &Vec<f64>) -> Vec<f64> {
        let (_, outputs) = self.forward(x);
        outputs
    }
}

impl Model for MLP {
    type Input = Vec<f64>;
    type Output = Vec<f64>;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        self.train(x, y, 1000);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        self.predict(x)
    }
}
