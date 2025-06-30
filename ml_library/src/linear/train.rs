use crate::linear::model::LinearModel;


pub fn train_linear(
    inputs: &[Vec<f64>],
    targets: &[f64],
    epochs: usize,
    learning_rate: f64,
) -> LinearModel {
    let mut model = LinearModel::new(inputs[0].len());
    for _ in 0..epochs {
        for (x, &y_true) in inputs.iter().zip(targets.iter()) {
            let y_pred = model.predict(x);
            let error  = y_true - y_pred;
            model.weights[0] += learning_rate * error;
            for i in 0..x.len() {
                model.weights[i + 1] += learning_rate * error * x[i];
            }
        }
    }
    model
}

