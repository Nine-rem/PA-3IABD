use iabd2_pa::models::linear_classification::LinearClassification;
use iabd2_pa::models::model_trait::Model;

#[test]
fn test_linear_classification_predict_basic() {
    let mut model = LinearClassification::new();
    model.set_weights(vec![0.5, 1.0, -1.0]); // biais + poids
    let input_pos = vec![2.0, 1.0];
    let input_neg = vec![1.0, 3.0];
    assert_eq!(model.predict(&input_pos), 1);
    assert_eq!(model.predict(&input_neg), -1);
}

#[test]
fn test_linear_classification_train_converges() {
    let mut model = LinearClassification::new();
    let x = vec![
        vec![2.0, 3.0],
        vec![1.0, 1.0],
        vec![4.0, 5.0],
        vec![5.0, 2.0],
    ];
    let y = vec![1, 1, -1, -1];
    model.train(&x, &y);
    for (xi, yi) in x.iter().zip(y.iter()) {
        assert_eq!(model.predict(xi), *yi);
    }
}
