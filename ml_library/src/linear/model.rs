
use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::SVD;

/// Modèle linéaire : classification binaire, multi‑classe ou régression
pub struct LinearModel {
    pub weights: DMatrix<f64>,  // 1×(d+1) ou K×(d+1)
}

impl LinearModel {
    /// Perceptron binaire de Rosenblatt
    pub fn perceptron_binary(
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        lr: f64,
        n_iters: usize,
    ) -> Self {
        let (n_samples, n_features) = x.shape();
        // x_aug : biais + x
        let mut x_aug = DMatrix::zeros(n_samples, n_features + 1);
        for i in 0..n_samples {
            x_aug[(i, 0)] = 1.0;
            for j in 0..n_features {
                x_aug[(i, j + 1)] = x[(i, j)];
            }
        }
        // poids initiaux
        let mut w = DVector::zeros(n_features + 1);

        for _ in 0..n_iters {
            for i in 0..n_samples {
                let xi = x_aug.row(i).transpose();
                let yi = y[i];
                let pred = if w.dot(&xi) >= 0.0 { 1.0 } else { -1.0 };
                let err = yi - pred;
                if err.abs() > 0.0 {
                    for k in 0..(n_features + 1) {
                        w[k] += lr * err * xi[k];
                    }
                }
            }
        }
        let w_matrix = DMatrix::from_row_slice(1, n_features + 1, w.as_slice());
        LinearModel { weights: w_matrix }
    }

    /// Perceptron multi‑classe (one‑vs‑rest), sans utiliser gemv
    pub fn perceptron_multiclass(
        x: &DMatrix<f64>,
        y_onehot: &DMatrix<f64>,
        lr: f64,
        n_iters: usize,
    ) -> Self {
        let (n_samples, n_features) = x.shape();
        let n_classes = y_onehot.ncols();
        // x_aug
        let mut x_aug = DMatrix::zeros(n_samples, n_features + 1);
        for i in 0..n_samples {
            x_aug[(i, 0)] = 1.0;
            for j in 0..n_features {
                x_aug[(i, j + 1)] = x[(i, j)];
            }
        }
        // poids initiaux
        let mut w_mat = DMatrix::zeros(n_classes, n_features + 1);

        for _ in 0..n_iters {
            for i in 0..n_samples {
                let xi = x_aug.row(i).transpose();
                // calcul scores manuellement
                let mut pred_class = 0;
                let mut max_score = std::f64::NEG_INFINITY;
                for c in 0..n_classes {
                    let mut score = 0.0;
                    for j in 0..(n_features + 1) {
                        score += w_mat[(c, j)] * xi[j];
                    }
                    if score > max_score {
                        max_score = score;
                        pred_class = c;
                    }
                }
                // vraie classe
                let true_class = y_onehot.row(i)
                    .iter()
                    .position(|&v| (v - 1.0).abs() < 1e-8)
                    .unwrap();
                if pred_class != true_class {
                    // mise à jour Rosenblatt
                    for j in 0..(n_features + 1) {
                        w_mat[(true_class, j)] += lr * xi[j];
                        w_mat[(pred_class,  j)] -= lr * xi[j];
                    }
                }
            }
        }
        LinearModel { weights: w_mat }
    }

            /// Régression linéaire OLS via pseudo-inverse (SVD)
    pub fn regression_ols(x: &DMatrix<f64>, y: &DVector<f64>) -> Self {
        let (n_samples, n_features) = x.shape();
        // construire x_aug
        let mut x_aug = DMatrix::zeros(n_samples, n_features + 1);
        for i in 0..n_samples {
            x_aug[(i, 0)] = 1.0;
            for j in 0..n_features {
                x_aug[(i, j + 1)] = x[(i, j)];
            }
        }
        // SVD pour pseudo-inverse
        let svd = SVD::new(x_aug.clone(), true, true);
        let tol = 1e-12;
        let x_pinv = svd.pseudo_inverse(tol)
            .expect("Échec du calcul de la pseudo-inverse via SVD");
        // calcul des poids w = X⁺ y
        let w_vec = x_pinv * y;
        // stocker en matrice 1×(d+1)
        let w_matrix = DMatrix::from_row_slice(1, n_features + 1, w_vec.as_slice());
        LinearModel { weights: w_matrix }
    }
}
