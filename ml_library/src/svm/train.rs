use crate::svm::model::SVM;
use rand::{thread_rng, Rng};

/// Entraîne un SVM linéaire via Pegasos (SGD sur hinge loss)
pub fn train_svm(
    inputs: &[Vec<f64>],
    targets: &[f64],
    iterations: usize,
    learning_rate: f64,
    lambda: f64,
) -> SVM {
    assert_eq!(
        inputs.len(),
        targets.len(),
        "inputs.len() doit == targets.len()"
    );

    let n_features = inputs[0].len();
    let mut model = SVM::new_py(n_features, learning_rate, lambda);
    let mut rng = thread_rng();

    for _ in 0..iterations {
        let idx = rng.gen_range(0..inputs.len());
        let x = &inputs[idx];
        let y = targets[idx];
        let margin = y * model.decision_function(x.clone());

        let η = model.learning_rate;
        let λ = model.lambda;

        if margin < 1.0 {
            // w = (1 - ηλ) w + η y x
            // biais
            model.weights[0] = (1.0 - η * λ) * model.weights[0] + η * y;
            // poids
            for i in 0..n_features {
                let w_old = model.weights[i + 1];
                model.weights[i + 1] = (1.0 - η * λ) * w_old + η * y * x[i];
            }
        } else {
            // seul le terme de régularisation s'applique
            for w in model.weights.iter_mut() {
                *w *= 1.0 - η * λ;
            }
        }
    }

    model
}
