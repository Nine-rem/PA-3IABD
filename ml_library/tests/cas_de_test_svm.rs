// ml_library/tests/cas_de_test_svm.rs

use ml_library::svm::train::train_svm;

fn bin_label(score: f64) -> f64 {
    if score >= 0.0 { 1.0 } else { -1.0 }
}

#[test]
fn linear_simple() {
    let inputs = vec![vec![1.0,1.0], vec![2.0,3.0], vec![3.0,3.0]];
    let targets = vec![1.0, -1.0, -1.0];
    let model = train_svm(&inputs, &targets, 50_000, 0.01, 1e-3);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}

#[test]
fn linear_multiple() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        inputs.push(vec![rng.gen::<f64>()*0.9+1.0, rng.gen::<f64>()*0.9+1.0]);
        targets.push(1.0);
    }
    for _ in 0..50 {
        inputs.push(vec![rng.gen::<f64>()*0.9+2.0, rng.gen::<f64>()*0.9+2.0]);
        targets.push(-1.0);
    }
    let model = train_svm(&inputs, &targets, 100_000, 0.01, 1e-3);
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}

#[test]
fn xor() {
    let inputs = vec![
        vec![1.0,0.0], vec![0.0,1.0],
        vec![0.0,0.0], vec![1.0,1.0],
    ];
    let targets = vec![1.0, 1.0, -1.0, -1.0];
    let model = train_svm(&inputs, &targets, 100_000, 0.01, 1e-3);
    // FIXME : SVM linéaire ne résout pas le XOR
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}

#[test]
fn cross() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..500 {
        let x = rng.gen::<f64>()*2.0 - 1.0;
        let y = rng.gen::<f64>()*2.0 - 1.0;
        inputs.push(vec![x,y]);
        targets.push(if x.abs()<=0.3 || y.abs()<=0.3 { 1.0 } else { -1.0 });
    }
    let model = train_svm(&inputs, &targets, 200_000, 0.01, 1e-3);
    // FIXME : ne gère pas la frontière en croix
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}

#[test]
fn multi_linear_3() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..500 {
        let a = rng.gen::<f64>()*2.0 - 1.0;
        let b = rng.gen::<f64>()*2.0 - 1.0;
        let lab = if -a-b-0.5>0.0 && b<0.0 && a-b-0.5<0.0 {
            0.0
        } else if -a-b-0.5<0.0 && b>0.0 && a-b-0.5<0.0 {
            1.0
        } else {
            2.0
        };
        inputs.push(vec![a,b]);
        targets.push(lab);
    }
    let model = train_svm(&inputs, &targets, 200_000, 0.01, 1e-3);
    // FIXME : pas de multiclass out-of-the-box
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}

#[test]
fn multi_cross() {
    use rand::Rng;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let x = rng.gen::<f64>()*2.0 - 1.0;
        let y = rng.gen::<f64>()*2.0 - 1.0;
        let lab = if (x%0.5).abs()<=0.25 && (y%0.5).abs()>0.25 {
            0.0
        } else if (x%0.5).abs()>0.25 && (y%0.5).abs()<=0.25 {
            1.0
        } else {
            2.0
        };
        inputs.push(vec![x,y]);
        targets.push(lab);
    }
    let model = train_svm(&inputs, &targets, 200_000, 0.01, 1e-3);
    // FIXME : multiclass non supportée nativement
    for (x, &y) in inputs.iter().zip(targets.iter()) {
        assert_eq!(bin_label(model.predict(x)), y);
    }
}
