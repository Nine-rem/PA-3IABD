// src/linear/train.rs

use std::slice;
use nalgebra::{DMatrix, DVector};
use crate::linear::model::LinearModel;

/// Perceptron binaire exposé à ctypes.
#[no_mangle]
pub extern "C" fn train_perceptron_binary(
    ptr_x: *const f64,
    ptr_y: *const f64,
    n_samples: usize,
    n_features: usize,
    lr: f64,
    n_iters: usize,
    out_weights: *mut f64,
) {
    let x_slice = unsafe { slice::from_raw_parts(ptr_x, n_samples * n_features) };
    let y_slice = unsafe { slice::from_raw_parts(ptr_y, n_samples) };
    let x = DMatrix::from_row_slice(n_samples, n_features, x_slice);
    let y = DVector::from_row_slice(y_slice);
    let model = LinearModel::perceptron_binary(&x, &y, lr, n_iters);
    let w = model.weights.as_slice();
    unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), out_weights, w.len()); }
}

/// Perceptron multi‑classe exposé à ctypes.
#[no_mangle]
pub extern "C" fn train_perceptron_multiclass(
    ptr_x: *const f64,
    ptr_y: *const f64,
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    lr: f64,
    n_iters: usize,
    out_weights: *mut f64,
) {
    let x_slice = unsafe { slice::from_raw_parts(ptr_x, n_samples * n_features) };
    let y_slice = unsafe { slice::from_raw_parts(ptr_y, n_samples * n_classes) };
    let x = DMatrix::from_row_slice(n_samples, n_features, x_slice);
    let y_onehot = DMatrix::from_row_slice(n_samples, n_classes, y_slice);
    let model = LinearModel::perceptron_multiclass(&x, &y_onehot, lr, n_iters);
    let w = model.weights.as_slice();
    unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), out_weights, w.len()); }
}

/// Régression linéaire (OLS) exposée à ctypes.
#[no_mangle]
pub extern "C" fn train_regression(
    ptr_x: *const f64,
    ptr_y: *const f64,
    n_samples: usize,
    n_features: usize,
    out_weights: *mut f64,
) {
    let x_slice = unsafe { slice::from_raw_parts(ptr_x, n_samples * n_features) };
    let y_slice = unsafe { slice::from_raw_parts(ptr_y, n_samples) };
    let x = DMatrix::from_row_slice(n_samples, n_features, x_slice);
    let y = DVector::from_row_slice(y_slice);
    let model = LinearModel::regression_ols(&x, &y);
    let w = model.weights.as_slice();
    unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), out_weights, w.len()); }
}