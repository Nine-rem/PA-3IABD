// tests/toy.rs

use ml_library::linear::train::train_linear;
use ml_library::perceptron::train::train_perceptron;
use ml_library::pmc::train::train_mlp;
use ml_library::rbfn::train::train_rbfn;
use ml_library::svm::train::train_svm;
use nalgebra::DVector;

#[test]
fn toy_regression_linear() {
    // y = 2x
    let inputs  = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets = vec![2.0, 4.0, 6.0];
    let model   = train_linear(&inputs, &targets, 1_000, 0.01);
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        let y_hat = model.predict(x);
        assert!((y_hat - t).abs() < 1e-2, "got {}, expected {}", y_hat, t);
    }
}

#[test]
fn toy_regression_mlp() {
    // y = 2x
    let inputs  = vec![vec![1.0], vec![2.0], vec![3.0]];
    let targets_scalar = vec![2.0, 4.0, 6.0];
    // convertir en Vec<Vec<f64>>
    let targets = targets_scalar.iter().map(|&y| vec![y]).collect::<Vec<_>>();
    let mut model = train_mlp(&inputs, &targets, &[], 1, 5_000, 0.01);
    for (x, &t) in inputs.iter().zip(targets_scalar.iter()) {
        let y_hat = model.predict(x)[0];
        assert!((y_hat - t).abs() < 1e-1, "got {}, expected {}", y_hat, t);
    }
}

#[test]
fn toy_regression_rbfn() {
    // y = x + 1
    let inputs  = vec![vec![0.0], vec![1.0], vec![2.0]];
    let targets = vec![1.0, 2.0, 3.0];
    let rbfn = train_rbfn(&inputs, &targets, inputs.len(), 1.0, 5);
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        let y_hat = rbfn.predict(x);
        assert!((y_hat - t).abs() < 1e-6, "got {}, expected {}", y_hat, t);
    }
}

#[test]
fn toy_classification_perceptron() {
    // Classe -1 vs +1 sur x=0 et x=1
    let inputs  = vec![vec![0.0], vec![1.0]];
    let targets = vec![-1.0, 1.0];
    let p = train_perceptron(&inputs, &targets, 1_000, 0.1);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(p.predict(x), y, "x={:?}", x);
    }
}

#[test]
fn toy_classification_mlp() {
    // Classe -1 vs +1 sur x=0 et x=1 (MLP sans cachée)
    let inputs  = vec![vec![0.0], vec![1.0]];
    let targets_scalar = vec![-1.0, 1.0];
    let targets = targets_scalar.iter().map(|&y| vec![y]).collect::<Vec<_>>();
    let mut model = train_mlp(&inputs, &targets, &[], 1, 5_000, 0.1);
    for (x, &y) in inputs.iter().zip(targets_scalar.iter()) {
        let out = model.predict(x)[0];
        let sign = if out >= 0.0 { 1.0 } else { -1.0 };
        assert_eq!(sign, y, "x={:?}, raw={}", x, out);
    }
}

#[test]
fn toy_classification_svm() {
    // Classe -1 vs +1 sur x=0 et x=1
    let inputs  = vec![vec![0.0], vec![1.0]];
    let targets = vec![-1.0, 1.0];
    let svm = train_svm(&inputs, &targets, 1_000, 0.1, 0.01);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(svm.predict(x), y, "x={:?}", x);
    }
}
