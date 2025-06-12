use iabd2_pa::models::mlp::MLP;
use iabd2_pa::models::model_trait::Model;

#[test]
fn test_mlp_predict_output_length() {
    let model = MLP::new(&[2, 3, 1], 0.1);
    let input = vec![0.5, -0.5];
    let output = model.predict(&input);
    assert_eq!(output.len(), 1);
}

#[test]
fn test_mlp_train_improves_output() {
    let mut model = MLP::new(&[2, 3, 1], 0.1);
    let x_train = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    model.train(&x_train, &y_train, 1000);
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        let pred = model.predict(x);
        let diff = (pred[0] - y[0]).abs();
        assert!(diff < 0.5, "Prediction {:.3} far from target {:.3}", pred[0], y[0]);
    }
}
