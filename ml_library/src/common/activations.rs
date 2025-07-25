
/// Fonction sigmoïde.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Dérivée de la sigmoïde à partir de la sortie sigmoïde.
/// (Évite de recalculer `sigmoid(x)`).
pub fn sigmoid_derivative_from_output(s: f64) -> f64 {
    s * (1.0 - s)
}

/// Fonction tanh.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Dérivée de tanh (1 - tanh²(x)).
pub fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

/// Fonction ReLU.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Dérivée de ReLU.
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Fonction signe (perceptron).
pub fn sign(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

/// Fonction softmax sur un vecteur (stabilisée).
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
