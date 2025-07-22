use crate::pmc::model::PMC;
use rand::Rng;

/// Entraîne un perceptron multiclasses (one-vs-rest).
/// - `inputs`: vecteurs de caractéristiques
/// - `targets`: indices de classes (0..n_classes-1)
/// - `iterations`: nb d’itérations stochastiques
/// - `learning_rate`: pas d’apprentissage
/// Panique si dimensions invalides.
pub fn train_pmc(
    inputs: &[Vec<f64>],
    targets: &[usize],
    iterations: usize,
    learning_rate: f64,
) -> PMC {
    let n_samples = inputs.len();
    assert_eq!(
        n_samples,
        targets.len(),
        "train_pmc: inputs.len() ({}) doit == targets.len() ({})",
        n_samples,
        targets.len()
    );
    let n_features = inputs[0].len();
    for x in inputs {
        assert_eq!(
            x.len(),
            n_features,
            "train_pmc: chaque vecteur a {} features, trouvé {}",
            n_features,
            x.len()
        );
    }

    // Nombre de classes
    let n_classes = *targets.iter().max().unwrap_or(&0) + 1;
    let mut rng = rand::thread_rng();

    // Initialiser poids (biais + features) pour chaque classe
    let mut weights: Vec<Vec<f64>> = (0..n_classes)
        .map(|_| vec![0.0; n_features + 1])
        .collect();

    // Perceptron stochastique one-vs-rest
    for _ in 0..iterations {
        let i = rng.gen_range(0..n_samples);
        let x = &inputs[i];
        let y_true = targets[i];

        // Calcul scores
        let scores: Vec<f64> = weights
            .iter()
            .map(|w| w[0] + w[1..]
                .iter()
                .zip(x)
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>())
            .collect();

        // Prédiction = argmax
        let y_pred = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // Mise à jour si erreur
        if y_pred != y_true {
            for j in 0..=n_features {
                let xi = if j == 0 { 1.0 } else { x[j - 1] };
                weights[y_true][j] += learning_rate * xi;
                weights[y_pred][j] -= learning_rate * xi;
            }
        }
    }

    PMC { weights }
}
