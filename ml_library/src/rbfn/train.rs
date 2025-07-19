use crate::rbfn::model::RBFN;
use nalgebra::DVector;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::common::math::{sub_vectors, dot_product};

/// Applique K-means simple pour trouver les centres
pub fn kmeans(inputs: &[Vec<f64>], k: usize, iterations: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let mut centers: Vec<Vec<f64>> = inputs.choose_multiple(&mut rng, k).cloned().collect();

    for _ in 0..iterations {
        let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); k];

        for x in inputs {
            let idx = centers
                .iter()
                .enumerate()
                .min_by(|(_, c1), (_, c2)| {
                    // 1) calcul manuel avec common::math pour éviter les problèmes de zip
                    let diff1 = sub_vectors(x, c1);
                    let diff2 = sub_vectors(x, c2);
                    let d1 = dot_product(&diff1, &diff1);
                    let d2 = dot_product(&diff2, &diff2);
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap();
            clusters[idx].push(x.clone());
        }

        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let mut mean = vec![0.0; cluster[0].len()];
                for x in cluster {
                    for (m, &xi) in mean.iter_mut().zip(x.iter()) {
                        *m += xi;
                    }
                }
                for m in &mut mean {
                    *m /= cluster.len() as f64;
                }
                centers[i] = mean;
            }
        }
    }

    centers
}

/// Entraîne un RBFN
pub fn train_rbfn(
    inputs: &[Vec<f64>],
    targets: &[f64],
    k_centers: usize,
    gamma: f64,
    kmeans_iterations: usize,
) -> RBFN {
    assert_eq!(inputs.len(), targets.len(), "inputs.len() must == targets.len()");
    let centers = kmeans(inputs, k_centers, kmeans_iterations);
    let mut rbfn = RBFN::new(centers, gamma);

    let phi = rbfn.compute_phi(inputs);
    let y_vec = DVector::from_vec(targets.to_vec());
    rbfn.fit_weights(&phi, &y_vec);
    rbfn
}
