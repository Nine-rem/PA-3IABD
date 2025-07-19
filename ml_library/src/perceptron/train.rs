use crate::perceptron::model::Perceptron;
use rand::Rng;

/// Entraîne le Perceptron avec la règle de Rosenblatt (stochastique).
pub fn train_perceptron(
    inputs: &[Vec<f64>],
    targets: &[f64],
    iterations: usize,
    learning_rate: f64,
) -> Perceptron {
    // Vérifie les dimensions d’entrée
    assert_eq!(inputs.len(), targets.len(),
        "inputs.len() doit == targets.len()");
    let mut model = Perceptron::new(inputs[0].len(), learning_rate);
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {
        let idx = rng.gen_range(0..inputs.len());
        let x = &inputs[idx];
        let y_true = targets[idx];
        let y_pred = model.predict(x);
        let error = y_true - y_pred;

        if error != 0.0 {
            let lr = model.learning_rate;
            // biais
            model.weights[0] += lr * error; 
            // poids sur chaque feature
            for (i, xi) in x.iter().enumerate() {
                model.weights[i + 1] += lr * error * xi;
            }
        }
    }

    model
}
