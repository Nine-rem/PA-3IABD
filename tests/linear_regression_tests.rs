use iabd2_pa::models::linear_regression::LinearRegression;
use iabd2_pa::models::model_trait::Model;
#[test]
fn test_linear_regression_predict_basic() {
    let mut model = LinearRegression::new();
    model.set_weights(vec![1.0, 2.0, 3.0]); // biais + poids
    let input = vec![1.0, 2.0];
    let prediction = model.predict(&input);
    assert_eq!(prediction, 1.0 + 2.0 * 1.0 + 3.0 * 2.0);
}

#[test]
fn test_linear_regression_train_initializes_weights() {
    let mut model = LinearRegression::new();
    let x = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let y = vec![1.0, 2.0, 3.0];
    model.train(&x, &y);
    assert_eq!(model.get_weights().len(), 3);
}
