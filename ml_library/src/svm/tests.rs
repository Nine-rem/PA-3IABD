#[cfg(test)]
mod tests {
    use crate::svm::model::SVM;
    use crate::svm::train::train_svm;

    #[test]
    fn test_decision_function() {
        let mut svm = SVM::new(2, 0.1, 0.01);
        svm.weights = vec![1.0, 2.0, -1.0];
        let v = svm.decision_function(&[1.0, 2.0]);
        assert!((v - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_predict_sign() {
        let mut svm = SVM::new(2, 0.1, 0.01);
        svm.weights = vec![0.0, 1.0, 0.0];
        assert_eq!(svm.predict(&[1.0, 0.0]), 1.0);
        assert_eq!(svm.predict(&[-1.0, 0.0]), -1.0);
    }

    #[test]
    fn test_train_svm_simple_separable() {
        let inputs = vec![vec![1.0], vec![-1.0]];
        let targets = vec![1.0, -1.0];
        let svm = train_svm(&inputs, &targets, 1_000, 0.1, 0.01);
        for (x, &y) in inputs.iter().zip(targets.iter()) {
            assert_eq!(svm.predict(x), y);
        }
    }

    #[test]
    #[should_panic]
    fn test_predict_dim_mismatch() {
        let svm = SVM::new(2, 0.1, 0.01);
        let _ = svm.predict(&[1.0]);
    }

    #[test]
    #[should_panic]
    fn test_train_dim_mismatch() {
        let inputs = vec![vec![1.0, 2.0]];
        let targets = vec![];
        let _ = train_svm(&inputs, &targets, 10, 0.1, 0.01);
    }


    // Les tests fonctionnels (end-to-end)

    #[test]
    fn functional_svm_train_and_predict() {
        // Séparable : y = sign(x)
        let inputs  = vec![vec![0.5], vec![-0.5], vec![2.0], vec![-2.0]];
        let targets = vec![1.0, -1.0, 1.0, -1.0];
        let svm = train_svm(&inputs, &targets, 5_000, 0.1, 0.01);
        for (x, &y) in inputs.iter().zip(targets.iter()) {
            assert_eq!(svm.predict(x), y);
        }
    }

    //Les tests de performance / convergence

    #[cfg(test)]
    mod tests {
    use crate::common::loss::hinge_loss;
    use crate::svm::train::train_svm;

    #[test]
    fn performance_svm_hinge_decrease() {
        // Séparable simple
        let inputs  = vec![vec![1.0], vec![-1.0]];
        let targets = vec![1.0, -1.0];

        // Hinge initial (tous scores = 0)
        let zero_scores  = vec![0.0; targets.len()];
        let initial_loss = hinge_loss(&targets, &zero_scores);

        // Entraînement
        let svm          = train_svm(&inputs, &targets, 1_000, 0.1, 0.01);
        let trained_scores = inputs.iter().map(|x| svm.decision_function(x)).collect::<Vec<_>>();
        let trained_loss   = hinge_loss(&targets, &trained_scores);

        assert!(
            trained_loss < initial_loss,
            "SVM hinge loss should decrease ({} → {})",
            initial_loss,
            trained_loss
        );
    }
}


}

