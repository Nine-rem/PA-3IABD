use ml_library::perceptron::train::train_perceptron;
use rand::Rng;

/// Seuil de convergence pour la classification binaire
fn label_of(score: f64) -> f64 {
    if score >= 0.0 { 1.0 } else { -1.0 }
}

#[test]
fn linear_simple() {
    let inputs = vec![ vec![1.0,1.0], vec![2.0,3.0], vec![3.0,3.0] ];
    let targets = vec![ 1.0, -1.0, -1.0 ];
    let model = train_perceptron(&inputs, &targets, 10_000, 0.1);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(label_of(model.predict(x)), y);
    }
}

#[test]
fn linear_multiple() {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        inputs.push(vec![
            rng.gen::<f64>() * 0.9 + 1.0,
            rng.gen::<f64>() * 0.9 + 1.0,
        ]);
        targets.push(1.0);
    }
    for _ in 0..50 {
        inputs.push(vec![
            rng.gen::<f64>() * 0.9 + 2.0,
            rng.gen::<f64>() * 0.9 + 2.0,
        ]);
        targets.push(-1.0);
    }
    let model = train_perceptron(&inputs, &targets, 20_000, 0.1);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(label_of(model.predict(x)), y);
    }
}

// XOR : on attend que ça panique
#[test]
#[should_panic(expected = "XOR incorrect")]
fn xor() {
    let inputs = vec![
        vec![1.0,0.0], vec![0.0,1.0],
        vec![0.0,0.0], vec![1.0,1.0],
    ];
    let targets = vec![ 1.0, 1.0, -1.0, -1.0 ];
    let model = train_perceptron(&inputs, &targets, 10_000, 0.1);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        let pred = model.predict(x);
        assert_eq!(
            label_of(pred), y,
            "XOR incorrect: {:?} -> {:?}", x, pred
        );
    }
}

// Frontière en croix : on attend que ça panique
#[test]
#[should_panic(expected = "cross incorrect")]
fn cross() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..500 {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;
        inputs.push(vec![x, y]);
        let label = if x.abs() <= 0.3 || y.abs() <= 0.3 { 1.0 } else { -1.0 };
        targets.push(label);
    }
    let model = train_perceptron(&inputs, &targets, 50_000, 0.05);
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        let pred = model.predict(x);
        assert_eq!(
            label_of(pred), t,
            "cross incorrect: {:?} -> {:?}", x, pred
        );
    }
}

// Multiclasses non supportées : on attend que ça panique
#[test]
#[should_panic(expected = "multi_linear_3 incorrect")]
fn multi_linear_3() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..500 {
        let a = rng.gen::<f64>() * 2.0 - 1.0;
        let b = rng.gen::<f64>() * 2.0 - 1.0;
        let label = if -a - b - 0.5 > 0.0 && b < 0.0 && a - b - 0.5 < 0.0 {
            0.0
        } else if -a - b - 0.5 < 0.0 && b > 0.0 && a - b - 0.5 < 0.0 {
            1.0
        } else {
            2.0
        };
        inputs.push(vec![a, b]);
        targets.push(label);
    }
    let model = train_perceptron(&inputs, &targets, 100_000, 0.05);
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        assert_eq!(
            label_of(model.predict(x)), t,
            "multi_linear_3 incorrect: {:?} -> {:?}", x, model.predict(x)
        );
    }
}

#[test]
#[should_panic(expected = "multi_cross incorrect")]
fn multi_cross() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;
        let label = if (x % 0.5).abs() <= 0.25 && (y % 0.5).abs() > 0.25 {
            0.0
        } else if (x % 0.5).abs() > 0.25 && (y % 0.5).abs() <= 0.25 {
            1.0
        } else {
            2.0
        };
        inputs.push(vec![x, y]);
        targets.push(label);
    }
    let model = train_perceptron(&inputs, &targets, 100_000, 0.05);
    for (x, &t) in inputs.iter().zip(targets.iter()) {
        assert_eq!(
            label_of(model.predict(x)), t,
            "multi_cross incorrect: {:?} -> {:?}", x, model.predict(x)
        );
    }
}
