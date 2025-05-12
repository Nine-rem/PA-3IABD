pub fn nonlinear_transform(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            row.iter()
                .flat_map(|v| vec![
                    *v,
                    v.powi(2),
                    v.sin(),
                ])
                .collect()
        })
        .collect()
}
