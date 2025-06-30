#[cfg(test)]
mod tests {
    use crate::perceptron::{model::Perceptron, train::train_perceptron};

    #[test]
    fn test_perceptron_simple() {
        let X = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
        let Y = vec![1.0, -1.0, -1.0];
        let mut model: Perceptron = train_perceptron(&X, &Y, 1000, 0.1);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    fn test_perceptron_multiple() {
        let mut X = Vec::new();
        let mut Y = Vec::new();
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 1.0,
                        rand::random::<f64>() * 0.9 + 1.0]);
            Y.push(1.0);
        }
        for _ in 0..50 {
            X.push(vec![rand::random::<f64>() * 0.9 + 2.0,
                        rand::random::<f64>() * 0.9 + 2.0]);
            Y.push(-1.0);
        }
        let mut model: Perceptron = train_perceptron(&X, &Y, 1000, 0.1);
        for (x, &y_true) in X.iter().zip(Y.iter()) {
            let pred = model.predict(x);
            assert_eq!(pred, y_true);
        }
    }

    #[test]
    #[ignore]
    fn test_perceptron_xor() {
        // Non adapté: le perceptron Rosenblatt ne résout pas le XOR.
    }

    #[test]
    #[ignore]
    fn test_perceptron_cross() {
        // Non adapté: la structure en croix n'est pas linéairement séparable.
    }

    #[test]
    #[ignore]
    fn test_perceptron_multi_three_classes() {
        // Non adapté: perceptron binaire uniquement.
    }

    #[test]
    #[ignore]
    fn test_perceptron_multi_cross() {
        // Non adapté: perceptron binaire ne gère pas multi-croix.
    }
}
