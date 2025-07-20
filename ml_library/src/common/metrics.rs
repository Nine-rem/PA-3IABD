use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::wrap_pyfunction;

/// Taux de bonnes prédictions (accuracy).
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!(
            "Mismatched lengths in accuracy_score: {} vs {}",
            y_true.len(),
            y_pred.len()
        );
    }
    let mut correct = 0;
    for i in 0..y_true.len() {
        if y_true[i] == y_pred[i] {
            correct += 1;
        }
    }
    correct as f64 / y_true.len() as f64
}


pub fn precision_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!(
            "Mismatched lengths in precision_score: {} vs {}",
            y_true.len(),
            y_pred.len()
        );
    }
    let mut tp = 0;
    let mut fp = 0;
    for i in 0..y_true.len() {
        let t = y_true[i];
        let p = y_pred[i];
        if p == 1.0 {
            if t == 1.0 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}


pub fn recall_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!(
            "Mismatched lengths in recall_score: {} vs {}",
            y_true.len(),
            y_pred.len()
        );
    }
    let mut tp = 0;
    let mut fn_ = 0;
    for i in 0..y_true.len() {
        let t = y_true[i];
        let p = y_pred[i];
        if t == 1.0 {
            if p == 1.0 {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    }
}

/// F1‐score pour la classe positive.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let prec = precision_score(y_true, y_pred);
    let rec = recall_score(y_true, y_pred);
    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}


#[pyfunction]
fn accuracy_score_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true.len() = {}, y_pred.len() = {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(accuracy_score(&y_true, &y_pred))
}

/// precision_score(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction]
fn precision_score_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true.len() = {}, y_pred.len() = {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(precision_score(&y_true, &y_pred))
}

/// recall_score(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction]
fn recall_score_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true.len() = {}, y_pred.len() = {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(recall_score(&y_true, &y_pred))
}

/// f1_score(y_true: List[float], y_pred: List[float]) -> float
#[pyfunction]
fn f1_score_py(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true.len() = {}, y_pred.len() = {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(f1_score(&y_true, &y_pred))
}

/// point d’entrée du sous‐module Python `common.metrics`
#[pymodule]
pub fn metrics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accuracy_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score_py, m)?)?;
    Ok(())
}
