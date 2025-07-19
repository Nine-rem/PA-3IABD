// src/loss.rs

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
              // d/dp [ -t ln p - (1-t) ln(1-p) ] = -(t/p) + ((1-t)/(1-p))
              let p = p.clamp(eps, 1.0 - eps);
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
