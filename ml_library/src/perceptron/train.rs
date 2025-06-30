use crate::perceptron::model::Perceptron;
use rand::Rng;

/// Entraîne le Perceptron avec la règle de Rosenblatt
pub fn train_perceptron(
    inputs: &[Vec<f64>],
    targets: &[f64],
    iterations: usize,
    learning_rate: f64,
) -> Perceptron {
    let mut model = Perceptron::new(inputs[0].len(), learning_rate);
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {

        let idx = rng.gen_range(0..inputs.len());
        let x = &inputs[idx];
        let y_true = targets[idx];
        let y_pred = model.predict(x);
        let error = y_true - y_pred;

        if error != 0.0 {
            model.weights[0] += learning_rate * error; 
            for (i, xi) in x.iter().enumerate() {
                model.weights[i + 1] += learning_rate * error * xi;
            }
        }
    }

    model
}


