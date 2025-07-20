// src/common/math.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::wrap_pyfunction;

/// Produit scalaire entre deux vecteurs.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Norme L2 d'un vecteur.
pub fn norm(a: &[f64]) -> f64 {
    dot_product(a, a).sqrt()
}

/// Addition de deux vecteurs.
pub fn add_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Soustraction de deux vecteurs.
pub fn sub_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Multiplication d’un vecteur par un scalaire.
pub fn scalar_mul(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|x| x * scalar).collect()
}

/// Produit élément par élément (Hadamard).
pub fn hadamard_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Produit matrice × vecteur.
pub fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot_product(row, vector)).collect()
}

/// Produit matrice × matrice.
pub fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if !b.is_empty() { b[0].len() } else { 0 };
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        assert_eq!(a[i].len(), b.len(), "Dimension mismatch for mat_mat_mul");
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..b.len() {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Matrice de Gram (φᵀ φ).
pub fn gram_matrix(phi: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = phi.len();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            g[i][j] = dot_product(&phi[i], &phi[j]);
        }
    }
    g
}

//
// Wrappers PyO3
//

#[pyfunction]
fn dot_product_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "length mismatch: a.len() = {}, b.len() = {}",
            a.len(),
            b.len()
        )));
    }
    Ok(dot_product(&a, &b))
}

#[pyfunction]
fn norm_py(a: Vec<f64>) -> PyResult<f64> {
    Ok(norm(&a))
}

#[pyfunction]
fn add_vectors_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("length mismatch between a and b"));
    }
    Ok(add_vectors(&a, &b))
}

#[pyfunction]
fn sub_vectors_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("length mismatch between a and b"));
    }
    Ok(sub_vectors(&a, &b))
}

#[pyfunction]
fn scalar_mul_py(a: Vec<f64>, scalar: f64) -> PyResult<Vec<f64>> {
    Ok(scalar_mul(&a, scalar))
}

#[pyfunction]
fn hadamard_product_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("length mismatch between a and b"));
    }
    Ok(hadamard_product(&a, &b))
}

#[pyfunction]
fn mat_vec_mul_py(matrix: Vec<Vec<f64>>, vector: Vec<f64>) -> PyResult<Vec<f64>> {
    if !matrix.is_empty() && matrix[0].len() != vector.len() {
        return Err(PyValueError::new_err("matrix columns must equal vector length"));
    }
    Ok(mat_vec_mul(&matrix, &vector))
}

#[pyfunction]
fn mat_mat_mul_py(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    if a[0].len() != b.len() {
        return Err(PyValueError::new_err("inner dimensions of a and b must match"));
    }
    Ok(mat_mat_mul(&a, &b))
}

#[pyfunction]
fn gram_matrix_py(phi: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    Ok(gram_matrix(&phi))
}

/// point d’entrée du sous-module Python `common.math`
#[pymodule]
pub fn math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot_product_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_py, m)?)?;
    m.add_function(wrap_pyfunction!(add_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(sub_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(hadamard_product_py, m)?)?;
    m.add_function(wrap_pyfunction!(mat_vec_mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(mat_mat_mul_py, m)?)?;
    m.add_function(wrap_pyfunction!(gram_matrix_py, m)?)?;
    Ok(())
}
