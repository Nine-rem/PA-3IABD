
use ml_library::rbfn::train::train_rbfn;

/// Tolérance pour la régression non linéaire
const EPS: f64 = 1e-6;
fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

#[test]
fn linear_simple_2d() {
    let inputs = vec![vec![1.0], vec![2.0]];
    let targets = vec![2.0, 3.0];
    let model = train_rbfn(&inputs, &targets, /*k=*/2, /*γ=*/1.0, /*km_it=*/10);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx(preds[0], 2.0) && approx(preds[1], 3.0));
}

#[test]
fn non_linear_simple_2d() {
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![2.0, 3.0, 2.5];
    let model = train_rbfn(&inputs, &targets, 3, 2.0, 10);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx(preds[0], 2.0)
         && approx(preds[1], 3.0)
         && approx(preds[2], 2.5));
}

#[test]
fn linear_simple_3d() {
    let inputs = vec![
        vec![1.0,1.0],
        vec![2.0,2.0],
        vec![3.0,1.0],
    ];
    let targets = vec![2.0, 3.0, 2.5];
    let model = train_rbfn(&inputs, &targets, 3, 1.0, 10);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx(preds[0], 2.0)
         && approx(preds[1], 3.0)
         && approx(preds[2], 2.5));
}

#[test]
fn linear_tricky_3d() {
    let inputs = vec![
        vec![1.0,1.0],
        vec![2.0,2.0],
        vec![3.0,3.0],
    ];
    let targets = vec![1.0, 2.0, 3.0];
    let model = train_rbfn(&inputs, &targets, 3, 1.0, 10);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx(preds[0], 1.0)
         && approx(preds[1], 2.0)
         && approx(preds[2], 3.0));
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
    let model = train_rbfn(&inputs, &targets, /*k=*/4, /*γ=*/5.0, /*km_it=*/20);
    let preds: Vec<f64> = inputs.iter().map(|x| model.predict(x)).collect();
    assert!(approx(preds[0], 2.0)
         && approx(preds[1], 1.0)
         && approx(preds[2], -2.0)
         && approx(preds[3], -1.0));
}
