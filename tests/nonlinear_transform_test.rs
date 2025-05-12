use pa_3a_iabd2::linear_regression::nonlinear_transform;

#[test]
fn nonlinear_transform_should_apply_x_squared_and_sin() {
    let input = vec![vec![1.0, 2.0]];
    let result = nonlinear_transform(&input);

    let expected = vec![
        1.0, 1.0, 1.0f64.sin(),  // pour 1.0
        2.0, 4.0, 2.0f64.sin(),  // pour 2.0
    ];

    for (i, (a, b)) in result[0].iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "index {i} → attendu {:.6}, obtenu {:.6}, diff {:.6}",
            b, a, diff
        );
    }
}
