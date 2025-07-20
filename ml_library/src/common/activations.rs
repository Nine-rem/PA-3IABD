use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Fonction sigmoïde.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Dérivée de la sigmoïde à partir de la sortie sigmoïde.
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

// —————— Wrappers PyO3 ——————

/// sigmoïde(x)
#[pyfunction(name = "sigmoid")]
fn sigmoid_py(x: f64) -> PyResult<f64> {
    Ok(sigmoid(x))
}

/// sigmoid_derivative_from_output(s)
#[pyfunction(name = "sigmoid_derivative_from_output")]
fn sigmoid_derivative_from_output_py(s: f64) -> PyResult<f64> {
    Ok(sigmoid_derivative_from_output(s))
}

/// tanh(x)
#[pyfunction(name = "tanh")]
fn tanh_py(x: f64) -> PyResult<f64> {
    Ok(tanh(x))
}

/// tanh_derivative(x)
#[pyfunction(name = "tanh_derivative")]
fn tanh_derivative_py(x: f64) -> PyResult<f64> {
    Ok(tanh_derivative(x))
}

/// relu(x)
#[pyfunction(name = "relu")]
fn relu_py(x: f64) -> PyResult<f64> {
    Ok(relu(x))
}

/// relu_derivative(x)
#[pyfunction(name = "relu_derivative")]
fn relu_derivative_py(x: f64) -> PyResult<f64> {
    Ok(relu_derivative(x))
}

/// sign(x)
#[pyfunction(name = "sign")]
fn sign_py(x: f64) -> PyResult<f64> {
    Ok(sign(x))
}

/// softmax(list[float])
#[pyfunction(name = "softmax")]
fn softmax_py(xs: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(softmax(&xs))
}

/// point d’entrée du sous-module Python `common.activations`
#[pymodule]
pub fn activations(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigmoid_py, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_derivative_from_output_py, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_py, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_derivative_py, m)?)?;
    m.add_function(wrap_pyfunction!(relu_py, m)?)?;
    m.add_function(wrap_pyfunction!(relu_derivative_py, m)?)?;
    m.add_function(wrap_pyfunction!(sign_py, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_py, m)?)?;
    Ok(())
}
