// src/perceptron/tests.rs

#[cfg(test)]
mod tests {
    use crate::perceptron::model::Perceptron;
    use crate::perceptron::train::train_perceptron;

    #[test]
    fn test_perceptron_predict() {
        let p = Perceptron {
            weights: vec![0.5, 0.5, -1.0],
            learning_rate: 0.1,
        };
        assert_eq!(p.predict(&[1.0, 1.0]), 1.0);
        assert_eq!(p.predict(&[-10.0, 10.0]), -1.0);
    }

    #[test]
    fn test_perceptron_decision_boundary() {
        let p = Perceptron {
            weights: vec![0.0, 1.0, -1.0],
            learning_rate: 0.1,
        };
        assert_eq!(p.predict(&[3.0, 2.0]), 1.0);
        assert_eq!(p.predict(&[2.0, 3.0]), -1.0);
    }

    #[test]
    fn test_train_perceptron_simple_or() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![-1.0, 1.0, 1.0, 1.0];
        let p = train_perceptron(&inputs, &targets, 1_000, 0.1);
        for (x, &y) in inputs.iter().zip(targets.iter()) {
            assert_eq!(p.predict(x), y);
        }
    }

    #[test]
    #[should_panic]
    fn test_predict_dim_mismatch() {
        let p = Perceptron::new(2, 0.1);
        let _ = p.predict(&[1.0]); // mauvaise dimension => panic
    }

    #[test]
    #[should_panic]
    fn test_train_dim_mismatch() {
        let inputs = vec![vec![0.0, 1.0]];
        let targets = vec![];
        let _ = train_perceptron(&inputs, &targets, 10, 0.1);
    }

    // End-to-end : OR doit être parfaitement appris
    #[test]
    fn functional_perceptron_train_and_predict() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![-1.0, 1.0, 1.0, 1.0];
        let p = train_perceptron(&inputs, &targets, 5_000, 0.1);
        let mut correct = 0;
        for (x, &y) in inputs.iter().zip(targets.iter()) {
            if p.predict(x) == y {
                correct += 1;
            }
        }
        assert_eq!(correct, inputs.len(), "Le Perceptron OR n'est pas parfaitement appris");
    }
}
