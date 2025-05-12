pub struct MLP {
    layers: Vec<Layer>,
}

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    outputs: Vec<f64>,
    inputs: Vec<f64>,
}

fn simple_random(seed: u64, index: usize) -> f64 {
    let prime = 31u64;
    let mut value = seed.wrapping_mul((index as u64 + 1) * prime);
    value = value.wrapping_add(12345);
    (value % 1000) as f64 / 1000.0 * 2.0 - 1.0  // valeur entre -1 et 1
}

impl Layer {
    fn new(input_size: usize, output_size: usize, seed: u64) -> Self {
        let weights = (0..output_size)
            .map(|i| {
                (0..input_size)
                    .map(|j| simple_random(seed, i * input_size + j))
                    .collect()
            })
            .collect();

        let biases = (0..output_size)
            .map(|i| simple_random(seed, 10_000 + i))
            .collect();

        Layer {
            weights,
            biases,
            outputs: vec![],
            inputs: vec![],
        }
    }

    fn activate(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn activate_derivative(a: f64) -> f64 {
        a * (1.0 - a)
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.inputs = input.clone();
        self.outputs = self
            .weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| {
                let z = w_row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f64>() + b;
                Self::activate(z)
            })
            .collect();
        self.outputs.clone()
    }

    fn backward(&mut self, dvalues: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut dinputs = vec![0.0; self.inputs.len()];
        for i in 0..self.outputs.len() {
            let dz = dvalues[i] * Self::activate_derivative(self.outputs[i]);
            for j in 0..self.inputs.len() {
                dinputs[j] += dz * self.weights[i][j];
                self.weights[i][j] -= learning_rate * dz * self.inputs[j];
            }
            self.biases[i] -= learning_rate * dz;
        }
        dinputs
    }
}

impl MLP {
    pub fn new(sizes: &[usize]) -> Self {
        let seed = 42;
        let layers = sizes.windows(2)
            .enumerate()
            .map(|(i, w)| Layer::new(w[0], w[1], seed + i as u64))
            .collect();
        MLP { layers }
    }

    pub fn predict(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut out = input.clone();
        for layer in self.layers.iter_mut() {
            out = layer.forward(&out);
        }
        out
    }

    pub fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>, learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                let output = self.predict(xi);
                let loss: f64 = output.iter().zip(yi.iter()).map(|(o, t)| (o - t).powi(2)).sum();
                total_loss += loss;

                let mut delta: Vec<f64> = output.iter().zip(yi.iter()).map(|(o, t)| 2.0 * (o - t)).collect();
                for layer in self.layers.iter_mut().rev() {
                    delta = layer.backward(&delta, learning_rate);
                }
            }
            if epoch % 50 == 0 {
                println!("Epoch {}, Loss {}", epoch + 1, total_loss / x.len() as f64);
            }
        }
    }
}
