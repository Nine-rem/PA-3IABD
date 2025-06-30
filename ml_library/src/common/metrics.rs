/// metrics.rs
/// Fonctions d’évaluation pour la classification et la régression.

/// Calcule la précision (accuracy) pour la classification (binaire ou multi-classes)
/// à partir de labels entiers ou génériques.
pub fn accuracy<T: PartialEq>(preds: &[T], targets: &[T]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    let correct = preds.iter()
        .zip(targets.iter())
        .filter(|(p, t)| *p == *t)
        .count();

    correct as f64 / preds.len() as f64
}

/// Calcule la Mean Squared Error (MSE) pour la régression.
pub fn mse(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p - *t).powi(2))
        .sum::<f64>() / preds.len() as f64
}

/// Calcule la Mean Absolute Error (MAE) pour la régression.
pub fn mae(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p - *t).abs())
        .sum::<f64>() / preds.len() as f64
}

/// Calcule la Root Mean Squared Error (RMSE) pour la régression.
pub fn rmse(preds: &[f64], targets: &[f64]) -> f64 {
    mse(preds, targets).sqrt()
}

/// Calcule le coefficient de détermination R² pour la régression.
pub fn r2_score(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_res = preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| (*t - *p).powi(2))
        .sum::<f64>();
    let ss_tot = targets.iter()
        .map(|t| (*t - mean_target).powi(2))
        .sum::<f64>();

    1.0 - ss_res / ss_tot
}

/// Calcule la Mean Absolute Percentage Error (MAPE) pour la régression.
pub fn mape(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");
    assert!(targets.iter().all(|&t| t.abs() > 0.0), "targets ne doivent pas contenir de zéro pour MAPE");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| ((*p - *t).abs() / (*t).abs()))
        .sum::<f64>() * 100.0 / preds.len() as f64
}

/// Calcule la symmetric Mean Absolute Percentage Error (SMAPE) pour la régression.
pub fn smape(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let denom = (*p).abs() + (*t).abs();
            if denom == 0.0 {
                0.0
            } else {
                2.0 * (*p - *t).abs() / denom
            }
        })
        .sum::<f64>() * 100.0 / preds.len() as f64
}

/// Calcule l’explained variance (variance expliquée) pour la régression.
pub fn explained_variance_score(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    let mean_true = targets.iter().sum::<f64>() / targets.len() as f64;
    let var_true = targets.iter()
        .map(|t| (*t - mean_true).powi(2))
        .sum::<f64>();
    let var_res = preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| (*t - *p).powi(2))
        .sum::<f64>();

    1.0 - var_res / var_true
}

/// Calcule la précision (precision) pour la classification binaire (positif = 1.0).
pub fn precision(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");

    let tp = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 1.0 && *t == 1.0)
        .count() as f64;
    let fp = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 1.0 && *t != 1.0)
        .count() as f64;

    if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) }
}

/// Calcule le rappel (recall) pour la classification binaire (positif = 1.0).
pub fn recall(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");

    let tp = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 1.0 && *t == 1.0)
        .count() as f64;
    let fn_ = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p != 1.0 && *t == 1.0)
        .count() as f64;

    if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) }
}

/// Calcule le F1-score pour la classification binaire.
pub fn f1_score(preds: &[f64], targets: &[f64]) -> f64 {
    let p = precision(preds, targets);
    let r = recall(preds, targets);
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// Calcule le log loss (cross-entropy) pour la classification binaire.
pub fn log_loss(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    assert!(!preds.is_empty(), "pas de données à comparer");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let p_val = *p;
            let p_clamped = p_val.min(1.0 - 1e-15).max(1e-15);
            - (
                *t * p_clamped.ln()
                + (1.0 - *t) * (1.0 - p_clamped).ln()
              )
        })
        .sum::<f64>() / preds.len() as f64
}

/// Calcule l’AUC-ROC pour la classification binaire.
pub fn auc_roc(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");

    let mut pairs: Vec<(f64, f64)> = preds.iter().cloned().zip(targets.iter().cloned()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let positives = targets.iter().filter(|&&t| t == 1.0).count() as f64;
    let negatives = targets.iter().filter(|&&t| t != 1.0).count() as f64;

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_score = std::f64::NAN;
    let mut area = 0.0;
    let mut prev_fp_rate = 0.0;
    let mut prev_tp_rate = 0.0;

    for &(score, label) in &pairs {
        if score != prev_score {
            area += (fp/negatives - prev_fp_rate) * (tp/positives + prev_tp_rate) / 2.0;
            prev_fp_rate = fp/negatives;
            prev_tp_rate = tp/positives;
            prev_score = score;
        }
        if label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    area + (1.0 - prev_fp_rate) * (1.0 + prev_tp_rate) / 2.0
}

/// Calcule la matrice de confusion pour la classification multi-classes.
pub fn confusion_matrix(
    preds: &[usize],
    targets: &[usize],
    n_classes: usize,
) -> Vec<Vec<usize>> {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");
    let mut matrix = vec![vec![0; n_classes]; n_classes];
    for (&p, &t) in preds.iter().zip(targets.iter()) {
        matrix[t][p] += 1;
    }
    matrix
}

/// Calcule précision, rappel et F1 moyen (macro, micro et pondéré) pour multi-classes.
pub fn precision_recall_f1_multiclass(
    preds: &[usize],
    targets: &[usize],
    n_classes: usize,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let cm = confusion_matrix(preds, targets, n_classes);
    let mut support = vec![0usize; n_classes];
    let mut p = vec![0.0; n_classes];
    let mut r = vec![0.0; n_classes];
    let mut f1 = vec![0.0; n_classes];

    for c in 0..n_classes {
        let tp = cm[c][c] as f64;
        let fp: f64 = (0..n_classes).filter(|&j| j != c).map(|j| cm[j][c] as f64).sum();
        let fn_: f64 = (0..n_classes).filter(|&j| j != c).map(|j| cm[c][j] as f64).sum();
        support[c] = cm[c].iter().sum();
        p[c] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        r[c] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        f1[c] = if p[c] + r[c] > 0.0 { 2.0 * p[c] * r[c] / (p[c] + r[c]) } else { 0.0 };
    }

    let total_sup: usize = support.iter().sum();
    let macro_p = p.iter().sum::<f64>() / n_classes as f64;
    let macro_r = r.iter().sum::<f64>() / n_classes as f64;
    let macro_f1 = f1.iter().sum::<f64>() / n_classes as f64;

    let micro_tp: f64 = (0..n_classes).map(|c| cm[c][c] as f64).sum();
    let micro_fp: f64 = (0..n_classes)
        .map(|c| (0..n_classes).filter(|&j| j != c).map(|j| cm[j][c] as f64).sum::<f64>())
        .sum();
    let micro_fn: f64 = (0..n_classes)
        .map(|c| (0..n_classes).filter(|&j| j != c).map(|j| cm[c][j] as f64).sum::<f64>())
        .sum();

    let micro_p = if micro_tp + micro_fp > 0.0 { micro_tp / (micro_tp + micro_fp) } else { 0.0 };
    let micro_r = if micro_tp + micro_fn > 0.0 { micro_tp / (micro_tp + micro_fn) } else { 0.0 };
    let micro_f1 = if micro_p + micro_r > 0.0 { 2.0 * micro_p * micro_r / (micro_p + micro_r) } else { 0.0 };

    let weighted_p = (0..n_classes).map(|c| p[c] * support[c] as f64).sum::<f64>() / total_sup as f64;
    let weighted_r = (0..n_classes).map(|c| r[c] * support[c] as f64).sum::<f64>() / total_sup as f64;
    let weighted_f1 = (0..n_classes).map(|c| f1[c] * support[c] as f64).sum::<f64>() / total_sup as f64;

    (macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, weighted_p, weighted_r, weighted_f1)
}

/// Calcule la balanced accuracy (moyenne des rappels par classe).
pub fn balanced_accuracy(preds: &[usize], targets: &[usize], n_classes: usize) -> f64 {
    let cm = confusion_matrix(preds, targets, n_classes);
    let mut sum_r = 0.0;

    for c in 0..n_classes {
        let tp = cm[c][c] as f64;
        let fn_: f64 = (0..n_classes).filter(|&j| j != c).map(|j| cm[c][j] as f64).sum();
        sum_r += if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    }

    sum_r / n_classes as f64
}

/// Calcule le Matthews Correlation Coefficient (MCC) pour classification binaire.
pub fn matthews_corrcoef(preds: &[u8], targets: &[u8]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");

    let tp = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 1 && *t == 1)
        .count() as f64;
    let tn = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 0 && *t == 0)
        .count() as f64;
    let fp = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 1 && *t == 0)
        .count() as f64;
    let fn_ = preds.iter()
        .zip(targets.iter())
        .filter(|&(p, t)| *p == 0 && *t == 1)
        .count() as f64;

    let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if denom == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / denom }
}

/// Calcule le Brier score pour la calibration des probabilités (binaire).
pub fn brier_score(preds: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(preds.len(), targets.len(), "preds et targets doivent être de même longueur");

    preds.iter()
        .zip(targets.iter())
        .map(|(p, t)| (*p - *t).powi(2))
        .sum::<f64>() / preds.len() as f64
}
