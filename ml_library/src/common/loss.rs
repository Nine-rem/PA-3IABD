use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::wrap_pyfunction;

/// Erreur quadratique moyenne (MSE).
/// Utilisée pour la régression et comme fonction de coût dans un PMC.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(t, p)| (t - p).powi(2))
          .sum::<f64>() / y_true.len() as f64
}

/// Dérivée de la MSE par rapport aux prédictions.
pub fn mse_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    assert_eq!(y_true.len(), y_pred.len());
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(t, p)| 2.0 * (p - t) / y_true.len() as f64)
          .collect()
}

/// Binary Cross‐Entropy (BCE) pour classification binaire.
/// y_pred doit être dans (0,1).
pub fn binary_cross_entropy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let eps = 1e-15;
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(&t, &p)| {
              let p = p.clamp(eps, 1.0 - eps);
              - (t * p.ln() + (1.0 - t) * (1.0 - p).ln())
          })
          .sum::<f64>() / y_true.len() as f64
}

/// Dérivée de la BCE par rapport aux prédictions.
pub fn binary_cross_entropy_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    assert_eq!(y_true.len(), y_pred.len());
    let eps = 1e-15;
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(&t, &p)| {
              let p = p.clamp(eps, 1.0 - eps);
              // d/dp [-t ln p - (1-t) ln(1-p)] = -(t/p) + ((1-t)/(1-p))
              ( -t / p + (1.0 - t) / (1.0 - p) ) / (y_true.len() as f64)
          })
          .collect()
}

/// Hinge Loss pour SVM (labels en ±1).
pub fn hinge_loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(&t, &p)| {
              let margin = 1.0 - t * p;
              if margin > 0.0 { margin } else { 0.0 }
          })
          .sum::<f64>() / y_true.len() as f64
}

/// Dérivée de la Hinge Loss par rapport aux prédictions.
pub fn hinge_loss_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    assert_eq!(y_true.len(), y_pred.len());
    y_true.iter()
          .zip(y_pred.iter())
          .map(|(&t, &p)| {
              if 1.0 - t * p > 0.0 { -t / (y_true.len() as f64) } else { 0.0 }
          })
          .collect()
}

//
// Wrappers PyO3
//

/// mse(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction(name = "mse")]
fn mse_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true.len() = {}, y_pred.len() = {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(mse(&y_true, &y_pred))
}

/// mse_derivative(y_true: List[float], y_pred: List[float]) -> List[float]
#[pyfunction(name = "mse_derivative")]
fn mse_derivative_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<Vec<f64>> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err("length mismatch between y_true and y_pred"));
    }
    Ok(mse_derivative(&y_true, &y_pred))
}

/// binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction(name = "binary_cross_entropy")]
fn binary_cross_entropy_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err("length mismatch between y_true and y_pred"));
    }
    Ok(binary_cross_entropy(&y_true, &y_pred))
}

/// binary_cross_entropy_derivative(y_true: List[float], y_pred: List[float]) -> List[float]
#[pyfunction(name = "binary_cross_entropy_derivative")]
fn binary_cross_entropy_derivative_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<Vec<f64>> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err("length mismatch between y_true and y_pred"));
    }
    Ok(binary_cross_entropy_derivative(&y_true, &y_pred))
}

/// hinge_loss(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction(name = "hinge_loss")]
fn hinge_loss_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err("length mismatch between y_true and y_pred"));
    }
    Ok(hinge_loss(&y_true, &y_pred))
}

/// hinge_loss_derivative(y_true: List[float], y_pred: List[float]) -> List[float]
#[pyfunction(name = "hinge_loss_derivative")]
fn hinge_loss_derivative_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<Vec<f64>> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err("length mismatch between y_true and y_pred"));
    }
    Ok(hinge_loss_derivative(&y_true, &y_pred))
}

/// point d’entrée du sous-module Python `common.loss`
#[pymodule]
pub fn loss(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mse_py, m)?)?;
    m.add_function(wrap_pyfunction!(mse_derivative_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy_derivative_py, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_loss_py, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_loss_derivative_py, m)?)?;
    Ok(())
}
