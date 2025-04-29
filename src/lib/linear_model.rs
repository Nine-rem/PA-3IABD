pub struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
}

fn simple_random(seed: u64, index: usize) -> f64 {
    let prime = 31u64;
    let mut value = seed.wrapping_mul((index as u64 + 1) * prime);
    value = value.wrapping_add(12345);
    (value % 1000) as f64 / 1000.0
}

impl LinearModel {
    pub fn new(dim: usize)-> Self{ 
        let mut weights =  Vec::new();
        for i in 0..dim {
            weights.push(simple_random(42, i));
        }
        let bias = simple_random(42, dim + 1000);
        LinearModel {
            weights,
            bias,
        }
    }
    
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        let mut somme = 0.0;
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            somme += *w * *x;
        }
        somme += self.bias;
        return somme;
    }

    pub fn fit(&mut self, X:&Vec<Vec<f64>>,y: &Vec<f64>, learning_rate: f64, epochs: usize){
        
        let predicted_length = X.len();

        for epoch in 0..epochs {
           let mut total_loss = 0.0;
           for i in 0..predicted_length {
                let prediction = self.predict(&X[i]);
                let error = y[i] - prediction;
                for j in 0..self.weights.len() {
                    self.weights[j] += learning_rate * error * X[i][j];
                }
                self.bias += learning_rate * error;
                total_loss += error.powi(2); //convergence                
           } 
            // Afficher la perte à chaque époque 
            if epoch % 50 == 0 {
                let loss = total_loss / predicted_length as f64;
                println!("Epoch: {}, Loss: {}", epoch + 1, loss);
            }
        }
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }
}

