use iabd2_pa::models::svm::SVM;
use iabd2_pa::models::model_trait::Model;

#[test]
fn test_svm_predict_basic() {
    let mut model = SVM::new(2, 0.001, 500, 0.01);
    model.set_weights(vec![1.0, -1.0]);
    model.set_bias(0.5);
    let input_pos = vec![2.0, 1.0];
    let input_neg = vec![1.0, 3.0];
    assert_eq!(model.predict(&input_pos), 1);
    assert_eq!(model.predict(&input_neg), -1);
}

#[test]
fn test_svm_train_converges() {
    let mut model = SVM::new(2, 0.0005, 2000, 0.01);
    let x_train = vec![
        vec![2.0, 3.0],
        vec![1.0, 1.0],
        vec![4.0, 5.0],
        vec![5.0, 2.0],
    ];
    let y_train = vec![1, 1, -1, -1];
    model.train(&x_train, &y_train);
    let mut correct = 0;
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        if model.predict(x) == *y {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / y_train.len() as f64;
    assert!(accuracy >= 0.75, "Accuracy too low: {}", accuracy);
}
