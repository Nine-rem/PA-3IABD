use crate::linear::model::LinearModel;
use crate::common::loss::mse_derivative;

/// Entraîne un modèle linéaire par descente de gradient sur la MSE (batch).
pub fn train_linear(
    inputs: &[Vec<f64>],
    targets: &[f64],
    epochs: usize,
    learning_rate: f64,
) -> LinearModel {
    let n_features = inputs[0].len();
    let mut model = LinearModel::new(n_features);

    for _ in 0..epochs {
        // 1) Calcul des prédictions
        let preds: Vec<f64> = inputs.iter()
            .map(|x| model.predict(x))
            .collect();

        // 2) Dérivée de la MSE : ∂L/∂y_pred pour chaque échantillon
        let grads = mse_derivative(targets, &preds);

        // 3) Mise à jour du biais : -lr * ∑ᵢ ∂L/∂y_predᵢ
        let grad_b: f64 = grads.iter().sum();
        model.weights[0] -= learning_rate * grad_b;

        // 4) Mise à jour des poids : -lr * ∑ᵢ (∂L/∂y_predᵢ * xᵢⱼ)
        for j in 0..n_features {
            let grad_wj: f64 = inputs.iter()
                .zip(grads.iter())
                .map(|(x, &g)| g * x[j])
                .sum();
            model.weights[j + 1] -= learning_rate * grad_wj;
        }
    }

    model
}
