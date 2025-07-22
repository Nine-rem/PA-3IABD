/* 
use ml_library::linear::train::train_linear;

/// Tolérance pour comparer des f64 en régression
const EPS: f64 = 1e-6;
fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

#[test]
fn linear_simple_2d() {
    let inputs = vec![vec![1.0], vec![2.0]];
    let targets = vec![2.0, 3.0];
    let model = train_linear(&inputs, &targets, /*epochs=*/1000, /*lr=*/0.1);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx_eq(preds[0], 2.0) && approx_eq(preds[1], 3.0));
}

#[test]
fn non_linear_simple_2d() {
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![2.0, 3.0, 2.5];
    let model = train_linear(&inputs, &targets, 1000, 0.1);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    // on n’attend PAS de sur-apprentissage
    assert!(!approx_eq(preds[2], 2.5), "LinearModel a sur-appris la non-linéarité ?");
}

#[test]
fn linear_simple_3d() {
    let inputs = vec![vec![1.0,1.0], vec![2.0,2.0], vec![3.0,1.0]];
    let targets = vec![2.0, 3.0, 2.5];
    let model = train_linear(&inputs, &targets, 2000, 0.1);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx_eq(preds[0], 2.0)
         && approx_eq(preds[1], 3.0)
         && approx_eq(preds[2], 2.5));
}

#[test]
fn linear_tricky_3d() {
    let inputs = vec![vec![1.0,1.0], vec![2.0,2.0], vec![3.0,3.0]];
    let targets = vec![1.0, 2.0, 3.0];
    let model = train_linear(&inputs, &targets, 2000, 0.1);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx_eq(preds[0], 1.0)
         && approx_eq(preds[1], 2.0)
         && approx_eq(preds[2], 3.0));
}

#[test]
fn non_linear_simple_3d() {
    let inputs = vec![
        vec![1.0,0.0],
        vec![0.0,1.0],
        vec![1.0,1.0],
        vec![0.0,0.0],
    ];
    let targets = vec![2.0, 1.0, -2.0, -1.0];
    let model = train_linear(&inputs, &targets, 2000, 0.1);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    // on n’attend PAS de sur-apprentissage de la non-linéarité
    assert!(!approx_eq(preds[2], -2.0), "LinearModel a trop sur-appris ?!");
}
*/