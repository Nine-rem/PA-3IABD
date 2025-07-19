#[cfg(test)]
mod tests {
    use crate::rbfn::model::RBFN;
    use nalgebra::DVector;

    #[test]
    fn test_rbf_center_distance_zero() {
        let rbfn = RBFN::new(vec![vec![1.0, 2.0]], 1.0);
        let v = rbfn.rbf(&[1.0, 2.0], &[1.0, 2.0]);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rbf_far_distance() {
        let rbfn = RBFN::new(vec![vec![0.0, 0.0]], 1.0);
        let v = rbfn.rbf(&[3.0, 4.0], &[0.0, 0.0]);
        assert!(v < 1e-10);
    }

    #[test]
    fn test_compute_phi_dimensions() {
        let rbfn = RBFN::new(vec![vec![0.0, 0.0], vec![1.0, 1.0]], 1.0);
        let phi = rbfn.compute_phi(&vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
        assert_eq!(phi.nrows(), 2);
        assert_eq!(phi.ncols(), 2);
    }

    #[test]
    fn test_predict_after_fit() {
        let inputs = vec![vec![1.0], vec![2.0]];
        let targets = vec![3.0, 5.0];
        let mut rbfn = RBFN::new(inputs.clone(), 1.0);
        let phi = rbfn.compute_phi(&inputs);
        let y = DVector::from_vec(targets.clone());
        rbfn.fit_weights(&phi, &y);
        for (x, &t) in inputs.iter().zip(targets.iter()) {
            assert!((rbfn.predict(x) - t).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic]
    fn test_rbf_dim_mismatch() {
        let rbfn = RBFN::new(vec![vec![1.0, 2.0]], 1.0);
        let _ = rbfn.rbf(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn test_predict_dim_mismatch() {
        let rbfn = RBFN::new(vec![vec![1.0, 2.0]], 1.0);
        let _ = rbfn.predict(&[1.0]);
    }

    #[test]
    #[should_panic]
    fn test_fit_weights_dim_mismatch() {
        let mut rbfn = RBFN::new(vec![vec![1.0]], 1.0);
        let phi = rbfn.compute_phi(&vec![vec![1.0]]);
        let y = DVector::from_vec(vec![1.0, 2.0]);
        rbfn.fit_weights(&phi, &y);
    }


    // Les tests fonctionnels (end-to-end)

    #[test]
    fn functional_rbfn_train_and_predict_trivial() {
        // On choisit centres = points, donc W = targets
        let inputs  = vec![vec![0.0], vec![1.0], vec![2.0]];
        let targets = vec![1.0, 2.0, 3.0];
        let mut rbfn = RBFN::new(inputs.clone(), 5.0);
        let phi = rbfn.compute_phi(&inputs);
        let y   = DVector::from_vec(targets.clone());
        rbfn.fit_weights(&phi, &y);
        for (x, &t) in inputs.iter().zip(targets.iter()) {
            assert!((rbfn.predict(x) - t).abs() < 1e-6);
        }
    }


        //Les tests de performance / convergence


    #[cfg(test)]
    mod tests {
    use crate::common::loss::mse;
    use crate::rbfn::model::RBFN;
    use nalgebra::DVector;

    #[test]
    fn performance_rbfn_mse_decrease() {
        // Centres = points initiaux, y = x
        let inputs  = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![1.0, 2.0, 3.0];

        let mut rbfn = RBFN::new(inputs.clone(), 1.0);

        // MSE initial (prédiction = 0)
        let zero_preds   = vec![0.0; targets.len()];
        let initial_mse  = mse(&targets, &zero_preds);

        // Fit
        let phi          = rbfn.compute_phi(&inputs);
        let y_vec        = DVector::from_vec(targets.clone());
        rbfn.fit_weights(&phi, &y_vec);

        let trained_preds= inputs.iter().map(|x| rbfn.predict(x)).collect::<Vec<_>>();
        let trained_mse  = mse(&targets, &trained_preds);

        assert!(
            trained_mse < initial_mse,
            "RBFN MSE should decrease ({} → {})",
            initial_mse,
            trained_mse
        );
    }
}

}
