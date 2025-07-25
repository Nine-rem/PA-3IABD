//pub mod model_old;  // ARCHIVES 
//pub mod train_old;  // ARCHIVES 
pub mod tests;
pub mod model;  // Version avec ndarray - VERSION ACTUELLE
pub mod train;  // Version avec ndarray - VERSION ACTUELLE

// Utiliser la version ndarray (actuelle)
pub use model::PMC;
