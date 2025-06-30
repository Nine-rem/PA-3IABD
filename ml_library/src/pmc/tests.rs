// File: ml_library/src/pmc/tests.rs

#[cfg(test)]
mod tests {
    use crate::pmc::{mlp::MLP, train::train_mlp};

    fn run_classification(layers: &[usize], X: Vec<Vec<f64>>, Y: Vec<f64>) {
        let mut model: MLP = train_mlp(layers, &X, &Y, 2000, 0.1);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_mlp_simple() {
        let X = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
        let Y = vec![1.0, -1.0, -1.0];
        run_classification(&[2, 1], X, Y);
    }

    #[test]
    fn test_mlp_multiple() {
        let mut X = Vec::new(); let mut Y = Vec::new();
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 1.0,
                        rand::random::<f64>() * 0.9 + 1.0]); Y.push(1.0);
        }
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 2.0,
                        rand::random::<f64>() * 0.9 + 2.0]); Y.push(-1.0);
        }
        run_classification(&[2, 1], X, Y);
    }

    #[test]
    fn test_mlp_xor() {
        let X = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        let Y = vec![1.0, 1.0, -1.0, -1.0];
        run_classification(&[2, 2, 1], X, Y);
    }

    #[test]
    fn test_mlp_cross() {
        let X: Vec<Vec<f64>> = (0..500).map(|_| vec![rand::random::<f64>() * 2.0 - 1.0,
                                                       rand::random::<f64>() * 2.0 - 1.0]).collect();
        let Y: Vec<f64> = X.iter().map(|p|
            if p[0].abs() <= 0.3 || p[1].abs() <= 0.3 { 1.0 } else { -1.0 }).collect();
        run_classification(&[2, 4, 1], X, Y);
    }

    #[test]
    fn test_mlp_multi_three_classes() {
        let mut X = Vec::new(); let mut Y = Vec::new();
        for _ in 0..1000 {
            let p = vec![rand::random::<f64>() * 2.0 - 1.0, rand::random::<f64>() * 2.0 - 1.0];
            let y = if -p[0] - p[1] - 0.5 > 0.0 && p[1] < 0.0 && p[0] - p[1] - 0.5 < 0.0 {
                        vec![1.0, 0.0, 0.0]
                    } else if -p[0] - p[1] - 0.5 < 0.0 && p[1] > 0.0 && p[0] - p[1] - 0.5 < 0.0 {
                        vec![0.0, 1.0, 0.0]
                    } else if -p[0] - p[1] - 0.5 < 0.0 && p[1] < 0.0 && p[0] - p[1] - 0.5 > 0.0 {
                        vec![0.0, 0.0, 1.0]
                    } else { continue; };
            X.push(p); Y.extend(y.iter());
        }
        let mut model: MLP = train_mlp(&[2, 3], &X, &Y, 2000, 0.1);
        for (i, x) in X.iter().enumerate() {
            let pred = model.predict(x);
            let expected = &Y[3*i..3*i+3];
            assert_eq!(pred, expected);
        }
    }

    #[test]
    fn test_mlp_multi_cross() {
        let X: Vec<Vec<f64>> = (0..1000).map(|_| vec![rand::random::<f64>() * 2.0 - 1.0,
                                                         rand::random::<f64>() * 2.0 - 1.0]).collect();
        let Y: Vec<f64> = X.iter().flat_map(|p|
            if (p[0].abs() % 0.5) <= 0.25 && (p[1].abs() % 0.5) > 0.25 {
                vec![1.0, 0.0, 0.0]
            } else if (p[0].abs() % 0.5) > 0.25 && (p[1].abs() % 0.5) <= 0.25 {
                vec![0.0, 1.0, 0.0]
            } else { vec![0.0, 0.0, 1.0] }
        ).collect();
        let mut model: MLP = train_mlp(&[2, 4, 3], &X, &Y, 2000, 0.1);
        for (i, x) in X.iter().enumerate() {
            let pred = model.predict(x);
            let expected = &Y[3*i..3*i+3];
            assert_eq!(pred, expected);
        }
    }
}
