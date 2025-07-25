// src/lib.rs
//! ml_library crate: modules et Trait Model

pub mod common;
pub mod linear;
pub mod perceptron;
pub mod pmc;
pub mod rbfn;
pub mod svm;

/// Trait générique pour tous les modèles
pub trait Model {
    /// Entraînement avec données d'entraînement et validation
    fn fit(
        &mut self,
        x_train: &[Vec<f32>],
        y_train: &[f32],
        x_val: &[Vec<f32>],
        y_val: &[f32],
    );

    /// Prédiction: renvoie un vecteur de sorties
    fn predict(&self, x: &[Vec<f32>]) -> Vec<f32>;

    /// Sauvegarde dans un fichier (binaire ou texte)
    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;

    /// Chargement depuis un fichier
    fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}