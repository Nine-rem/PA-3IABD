#[cfg(test)]
mod tests {
    use crate::linear::model::LinearModel;
    use crate::linear::train::train_linear;

    #[test]
    fn test_linear_predict() {
        let model = LinearModel { weights: vec![0.5, 1.0, -0.5] };
        let x = vec![2.0, 4.0];
        assert!((model.predict(&x) - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_train_zero_targets() {
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![0.0, 0.0];
        let model = train_linear(&inputs, &targets, 10, 0.1);
        assert_eq!(model.weights, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_train_trivial() {
        // y = 2 * x
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![2.0, 4.0, 6.0];
        let model = train_linear(&inputs, &targets, 2_000, 0.01);
        // biais ≈ 0, pente ≈ 2
        assert!((model.weights[0]).abs() < 0.1);
        assert!((model.weights[1] - 2.0).abs() < 0.1);
    }

    #[test]
    #[should_panic]
    fn test_train_dim_mismatch() {
        let inputs = vec![vec![1.0]];
        let targets = vec![1.0, 2.0];
        let _ = train_linear(&inputs, &targets, 10, 0.1);
    }

    #[test]
    #[should_panic]
    fn test_predict_dim_mismatch() {
        let model = LinearModel::new(2);
        let _ = model.predict(&[1.0]); // input length != 2
    }

    //Les tests fonctionnels (end-to-end)

    #[test]
    fn functional_linear_train_and_predict() {
        // y = 2*x + 1
        let inputs  = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![1.0, 3.0, 5.0, 7.0];
        let model = train_linear(&inputs, &targets, 2_000, 0.01);
        for (x, &y) in inputs.iter().zip(targets.iter()) {
            let y_hat = model.predict(x);
            assert!((y_hat - y).abs() < 1e-2, "got {}, expected {}", y_hat, y);
        }
    }


    //Les tests de performance / convergence

    #[cfg(test)]
    mod tests {
        use crate::linear::train::train_linear;
        use crate::common::loss::mse;

        #[test]
        fn performance_linear_mse_decrease() {
            // y = 2x
            let inputs  = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
            let targets = vec![2.0, 4.0, 6.0, 8.0];

            // MSE initial (prédiction = 0)
            let zero_preds   = vec![0.0; targets.len()];
            let initial_mse  = mse(&targets, &zero_preds);

            // Entraînement
            let model        = train_linear(&inputs, &targets, 500, 0.01);
            let preds        = inputs.iter().map(|x| model.predict(x)).collect::<Vec<_>>();
            let trained_mse  = mse(&targets, &preds);

            assert!(
                trained_mse < initial_mse,
                "MSE should decrease, got {} >= {}",
                trained_mse,
                initial_mse
            );
        }
    }


}
