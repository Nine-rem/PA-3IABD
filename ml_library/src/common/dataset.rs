use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::wrap_pyfunction;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Charge un dataset CSV (sans header) en deux vecteurs : features (X) et labels (y).
/// Assume que la dernière colonne est le label.
pub fn load_csv_dataset(path: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(path).expect("Impossible d'ouvrir le fichier CSV.");
    let reader = BufReader::new(file);

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("Erreur de parsing"))
            .collect();

        let (features, label) = values.split_at(values.len() - 1);
        x_data.push(features.to_vec());
        y_data.push(label[0]);
    }

    (x_data, y_data)
}

/// Sépare le dataset en train/test selon un ratio (par ex. 0.8).
pub fn train_test_split<T: Clone>(
    data: &[T],
    labels: &[f64],
    ratio: f64,
) -> (Vec<T>, Vec<f64>, Vec<T>, Vec<f64>) {
    let train_size = (data.len() as f64 * ratio).round() as usize;

    let x_train = data[..train_size].to_vec();
    let y_train = labels[..train_size].to_vec();

    let x_test = data[train_size..].to_vec();
    let y_test = labels[train_size..].to_vec();

    (x_train, y_train, x_test, y_test)
}

//
// Wrappers PyO3
//

/// load_csv_dataset(path: str) -> (List[List[float]], List[float])
#[pyfunction]
fn load_csv_dataset_py(path: &str) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    // Propagate any panic as Python exception
    let (x, y) = load_csv_dataset(path);
    Ok((x, y))
}

/// train_test_split(
///     x: List[List[float]],
///     y: List[float],
///     ratio: float
/// ) -> (List[List[float]], List[float], List[List[float]], List[float])
#[pyfunction]
fn train_test_split_py(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    ratio: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err(format!(
            "mismatched lengths: x.len() = {}, y.len() = {}",
            x.len(),
            y.len()
        )));
    }
    let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, ratio);
    Ok((x_train, y_train, x_test, y_test))
}

/// point d’entrée du sous-module Python `common.dataset`
#[pymodule]
pub fn dataset(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_csv_dataset_py, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split_py, m)?)?;
    Ok(())
}
