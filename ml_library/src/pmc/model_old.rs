/*
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

pub type Matrix = DMatrix<f64>;
pub type Vector = DVector<f64>;
pub type Label = f64;
pub type Loss = f64;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON (de)serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Bincode (de)serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Dimension mismatch: {0}")]
    Dimension(String),
}

pub trait Model {
    type Err;

    /// Prédit une étiquette par ligne de `x`.
    fn predict(&self, x: &Matrix) -> Result<Vec<Label>, Self::Err>;

    /// Sérialisation lisible (JSON).
    fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Self::Err>;
    fn load_json<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Self::Err>;

    /// Sérialisation compacte (bincode).
    fn save_bincode<P: AsRef<Path>>(&self, path: P) -> Result<(), Self::Err>;
    fn load_bincode<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Self::Err>;

    /// Historiques.
    fn train_loss_history(&self) -> &[Loss];
    fn val_loss_history(&self) -> &[Loss];
    fn train_accuracy_history(&self) -> &[Loss];
    fn val_accuracy_history(&self) -> &[Loss];
}
*/