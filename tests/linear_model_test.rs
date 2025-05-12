use pa_3a_iabd2::linear_model::LinearModel;
use pa_3a_iabd2::linear_regression::nonlinear_transform;

#[test]
fn linear_model_should_learn_to_separate_classes() {
    // Jeu de données : 2 classes non-linéaires
    let x = vec![
        vec![1.0, 2.0],  // classe 0
        vec![2.0, 1.0],  // classe 1
        vec![2.0, 3.0],  // classe 1
        vec![0.5, 1.0],  // classe 0
    ];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let x_t = nonlinear_transform(&x);
    let mut model = LinearModel::new(x_t[0].len());
    model.fit(&x_t, &y, 0.1, 2000);

    // On vérifie que les prédictions sont du bon côté de la barrière 0.5
    let predictions: Vec<f64> = x_t.iter().map(|xi| model.predict(xi)).collect();

    assert!(predictions[0] < 0.5, "classe 0 mal classée (valeur = {})", predictions[0]);
    assert!(predictions[1] > 0.5, "classe 1 mal classée (valeur = {})", predictions[1]);
    assert!(predictions[2] > 0.5, "classe 1 mal classée (valeur = {})", predictions[2]);
    assert!(predictions[3] < 0.5, "classe 0 mal classée (valeur = {})", predictions[3]);
}
