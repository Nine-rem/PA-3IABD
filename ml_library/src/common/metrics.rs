/// Petite tolérance pour comparaison de flottants (plus réaliste que EPSILON).
const TOLERANCE: f64 = 1e-6;

/// Accuracy (taux de classification correcte).
/// Suppose que y_true et y_pred contiennent des classes 0.0 ou 1.0.
pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let correct = y_true
        .iter().copied()
        .zip(y_pred.iter().copied())
        .filter(|&(t, p)| (t - p).abs() < TOLERANCE)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Accuracy avec un seuil (utile pour sorties sigmoïdes).
pub fn accuracy_threshold(y_true: &[f64], y_pred: &[f64], threshold: f64) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let correct = y_true
        .iter().copied()
        .zip(y_pred.iter().copied())
        .filter(|&(t, p)| {
            let pred = if p >= threshold { 1.0 } else { 0.0 };
            (t - pred).abs() < TOLERANCE
        })
        .count();
    correct as f64 / y_true.len() as f64
}

/// Mean Squared Error (MSE).
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter().copied()
        .zip(y_pred.iter().copied())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

/// Mean Absolute Error (MAE).
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter().copied()
        .zip(y_pred.iter().copied())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64
}

/// Precision (TP / (TP + FP)).
pub fn precision(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let mut tp = 0;
    let mut fp = 0;
    for (t, p) in y_true.iter().copied().zip(y_pred.iter().copied()) {
        if p > 0.5 {
            if t > 0.5 {
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

/// Recall (TP / (TP + FN)).
pub fn recall(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let mut tp = 0;
    let mut fn_ = 0;
    for (t, p) in y_true.iter().copied().zip(y_pred.iter().copied()) {
        if t > 0.5 {
            if p > 0.5 {
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

/// F1-score = 2 * (Precision * Recall) / (Precision + Recall).
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let p = precision(y_true, y_pred);
    let r = recall(y_true, y_pred);
    if (p + r).abs() < TOLERANCE {
        0.0
    } else {
        2.0 * (p * r) / (p + r)
    }
}
