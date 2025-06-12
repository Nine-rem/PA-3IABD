use iabd2_pa::models::rbfn::RBFN;
use iabd2_pa::models::model_trait::Model;
#[test]
fn test_rbfn_train_and_predict() {
    let mut model = RBFN::new(3, 1.0, 10);
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
    ];
    let y_train = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    model.train(&x_train, &y_train);
    let x_test = vec![1.5, 1.5];
    let pred = model.predict(&x_test);
    assert!(pred.is_finite());
}
