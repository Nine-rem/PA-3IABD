// src/pmc/tests.rs
/*/
#[cfg(test)]
mod tests {
    use crate::pmc::model::pmc;
    use crate::pmc::train::train_pmc;

    /// Binarisation pour les sorties scalaires
    fn bin_label(score: f64) -> f64 {
        if score >= 0.0 { 1.0 } else { -1.0 }
    }

    /// Renvoie l’indice du max dans un slice
    fn argmax(xs: &[f64]) -> usize {
        xs.iter()
          .enumerate()
          .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
          .unwrap()
          .0
    }

    #[test]
    fn linear_simple() {
        let inputs = vec![ vec![1.0,1.0], vec![2.0,3.0], vec![3.0,3.0] ];
        let targets = vec![ vec![ 1.0 ], vec![ -1.0 ], vec![ -1.0 ] ];

        // pmc 2→1, lr=0.1
        let mut net: pmc = train_pmc(
            &inputs,
            &[2],    // une couche cachée de taille 2
            1,       // 1 neurone de sortie
            0.1,     // learning rate
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            assert_eq!(bin_label(out[0]), y[0], "linear_simple failed for {:?}", x);
        }
    }

    #[test]
    fn linear_multiple() {
        use rand::Rng;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            inputs.push(vec![ rng.gen::<f64>()*0.9 + 1.0,
                              rng.gen::<f64>()*0.9 + 1.0 ]);
            targets.push(vec![ 1.0 ]);
        }
        for _ in 0..50 {
            inputs.push(vec![ rng.gen::<f64>()*0.9 + 2.0,
                              rng.gen::<f64>()*0.9 + 2.0 ]);
            targets.push(vec![ -1.0 ]);
        }

        // pmc 2→1, lr=0.1
        let mut net: pmc = train_pmc(
            &inputs,
            &[2],    // une couche cachée de taille 2
            1,
            0.1,
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            assert_eq!(bin_label(out[0]), y[0], "linear_multiple failed for {:?}", x);
        }
    }

    #[test]
    fn xor() {
        let inputs = vec![
            vec![1.0,0.0], vec![0.0,1.0],
            vec![0.0,0.0], vec![1.0,1.0],
        ];
        let targets = vec![
            vec![ 1.0 ], vec![ 1.0 ], vec![ -1.0 ], vec![ -1.0 ]
        ];

        // pmc 2→2→1 pour résoudre XOR, lr=0.1
        let mut net: pmc = train_pmc(
            &inputs,
            &[2, 2],  // 2 neurones, puis 2 neurones
            1,
            0.1,
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            assert_eq!(
                bin_label(out[0]), y[0],
                "xor failed for {:?} -> {:?}", x, out
            );
        }
    }

    #[test]
    fn cross() {
        use rand::Rng;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..500 {
            let a = rng.gen::<f64>()*2.0 - 1.0;
            let b = rng.gen::<f64>()*2.0 - 1.0;
            inputs.push(vec![a,b]);
            targets.push(vec![ if a.abs()<=0.3 || b.abs()<=0.3 { 1.0 } else { -1.0 } ]);
        }

        // pmc 2→4→1 pour frontière en croix, lr=0.05
        let mut net: pmc = train_pmc(
            &inputs,
            &[4],
            1,
            0.05,
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            assert_eq!(
                bin_label(out[0]), y[0],
                "cross failed for {:?}", x
            );
        }
    }

    #[test]
    fn multi_linear_3() {
        use rand::Rng;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..500 {
            let a = rng.gen::<f64>()*2.0 - 1.0;
            let b = rng.gen::<f64>()*2.0 - 1.0;
            let onehot = if -a - b - 0.5 > 0.0 && b < 0.0 && a - b - 0.5 < 0.0 {
                vec![1.0,0.0,0.0]
            } else if -a - b - 0.5 < 0.0 && b > 0.0 && a - b - 0.5 < 0.0 {
                vec![0.0,1.0,0.0]
            } else {
                vec![0.0,0.0,1.0]
            };
            inputs.push(vec![a,b]);
            targets.push(onehot);
        }

        // pmc 2→3 pour 3 classes, lr=0.05
        let mut net: pmc = train_pmc(
            &inputs,
            &[3],
            3,
            0.05,
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            let p = argmax(&out);
            let t = argmax(&y);
            assert_eq!(
                p, t,
                "multi_linear_3 failed for {:?} -> {:?}", x, out
            );
        }
    }

    #[test]
    fn multi_cross() {
        use rand::Rng;
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<f64>()*2.0 - 1.0;
            let b = rng.gen::<f64>()*2.0 - 1.0;
            let onehot = if (a % 0.5).abs() <= 0.25 && (b % 0.5).abs() > 0.25 {
                vec![1.0,0.0,0.0]
            } else if (a % 0.5).abs() > 0.25 && (b % 0.5).abs() <= 0.25 {
                vec![0.0,1.0,0.0]
            } else {
                vec![0.0,0.0,1.0]
            };
            inputs.push(vec![a,b]);
            targets.push(onehot);
        }

        // pmc 2→5→3 pour cette topologie, lr=0.05
        let mut net: pmc = train_pmc(
            &inputs,
            &[5],
            3,
            0.05,
        );

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let out = net.predict(x);
            let p = argmax(&out);
            let t = argmax(&y);
            assert_eq!(
                p, t,
                "multi_cross failed for {:?} -> {:?}", x, out
            );
        }
    }
}
*/