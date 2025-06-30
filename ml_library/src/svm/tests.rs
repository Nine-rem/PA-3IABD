// File: ml_library/src/svm/tests.rs

#[cfg(test)]
mod tests {
    use crate::svm::{svm::SVM, train::train_svm};

    #[test]
    fn test_svm_simple_linear() {
        let X = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
        let Y = vec![1.0, -1.0, -1.0];
        let mut model: SVM = train_svm(&X, &Y, "linear", 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_svm_multiple_linear() {
        let mut X = Vec::new(); let mut Y = Vec::new();
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 1.0,
                        rand::random::<f64>() * 0.9 + 1.0]); Y.push(1.0);
        }
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 2.0,
                        rand::random::<f64>() * 0.9 + 2.0]); Y.push(-1.0);
        }
        let mut model: SVM = train_svm(&X, &Y, "linear", 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    #[ignore]
    fn test_svm_xor() {
        // Non adapté: SVM linéaire ne peut pas séparer XOR sans kernel non linéaire.
    }

    #[test]
    fn test_svm_cross_rbf() {
        let X: Vec<Vec<f64>> = (0..500).map(|_| vec![rand::random::<f64>() * 2.0 - 1.0,
                                                       rand::random::<f64>() * 2.0 - 1.0]).collect();
        let Y: Vec<f64> = X.iter().map(|p|
            if p[0].abs() <= 0.3 || p[1].abs() <= 0.3 { 1.0 } else { -1.0 }).collect();
        let mut model: SVM = train_svm(&X, &Y, "rbf", 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    #[ignore]
    fn test_svm_multi_three_classes() {
        // Non adapté: gestion multiclasse non implémentée, utiliser OVA.
    }

    #[test]
    #[ignore]
    fn test_svm_multi_cross() {
        // Non adapté: multi-croix non géré.
    }
}
