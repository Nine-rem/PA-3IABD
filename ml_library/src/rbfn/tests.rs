// File: ml_library/src/rbfn/tests.rs

#[cfg(test)]
mod tests {
    use crate::rbfn::{rbfn::RBFN, train::train_rbfn};

    #[test]
    fn test_rbfn_simple() {
        let X = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
        let Y = vec![1.0, -1.0, -1.0];
        let mut model: RBFN = train_rbfn(&X, &Y, X.len(), 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_rbfn_multiple() {
        let mut X = Vec::new(); let mut Y = Vec::new();
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 1.0,
                        rand::random::<f64>() * 0.9 + 1.0]); Y.push(1.0);
        }
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 2.0,
                        rand::random::<f64>() * 0.9 + 2.0]); Y.push(-1.0);
        }
        let mut model: RBFN = train_rbfn(&X, &Y, X.len(), 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_rbfn_xor() {
        let X = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        let Y = vec![1.0, 1.0, -1.0, -1.0];
        let mut model: RBFN = train_rbfn(&X, &Y, 4, 1.0);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_rbfn_cross() {
        let X: Vec<Vec<f64>> = (0..500).map(|_| vec![rand::random::<f64>() * 2.0 - 1.0,
                                                       rand::random::<f64>() * 2.0 - 1.0]).collect();
        let Y: Vec<f64> = X.iter().map(|p|
            if p[0].abs() <= 0.3 || p[1].abs() <= 0.3 { 1.0 } else { -1.0 }).collect();
        let mut model: RBFN = train_rbfn(&X, &Y, 10, 0.5);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    #[ignore]
    fn test_rbfn_multi_three_classes() {
        // Non adapté: implémentation RBFN binaire uniquement.
    }

    #[test]
    #[ignore]
    fn test_rbfn_multi_cross() {
        // Non adapté: gestion multi-croix non implémentée pour RBFN.
    }
}
