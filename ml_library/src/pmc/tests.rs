// src/pmc/tests.rs

#[cfg(test)]
mod tests {
    use crate::common::loss::mse;
    use crate::pmc::train::train_mlp;

    // ... vos autres tests ici ...

    #[test]
    fn performance_mlp_mse_decrease() {
        // y = 2x, réseau 1→2→1
        let inputs  = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![2.0, 4.0, 6.0, 8.0];

        // MSE initial (tous zéros)
        let zero_outs    = vec![vec![0.0]; inputs.len()];
        let initial_pred = zero_outs.iter().map(|v| v[0]).collect::<Vec<_>>();
        let initial_mse  = mse(&targets, &initial_pred);

        // Entraînement
        let mut model = train_mlp(
            &inputs,
            &targets.iter().map(|&y| vec![y]).collect::<Vec<_>>(),
            &[2],
            1,
            5_000,
            0.01,
        );

        // MSE après entraînement
        let trained_preds = inputs
            .iter()
            .map(|x| model.predict(x)[0])
            .collect::<Vec<_>>();
        let trained_mse   = mse(&targets, &trained_preds);

        assert!(
            trained_mse < initial_mse,
            "MLP MSE should decrease ({} → {})",
            initial_mse,
            trained_mse
        );
    }
}
