use crate::models::model_trait::Model;
use crate::models::utils::{transpose, matmul, matvecmul, inverse, squared_euclidean_distance, mean_vector};
use rand::Rng;


pub struct RBFN {
    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
    k: usize,
    max_kmeans_iter: usize,
}

impl RBFN {
    pub fn new(k: usize, gamma: f64, max_kmeans_iter: usize) -> Self {
        RBFN {
            centers: Vec::new(),
            weights: Vec::new(),
            gamma,
            k,
            max_kmeans_iter,
        }
    }

    fn gaussian_rbf(&self, x: &Vec<f64>, c: &Vec<f64>) -> f64 {
        let dist_sq: f64 = x.iter().zip(c.iter()).map(|(xi, ci)| (xi - ci).powi(2)).sum();
        (-self.gamma * dist_sq).exp()
    }
}

impl Model for RBFN {
    type Input = Vec<f64>;
    type Output = f64;

    fn train(&mut self, x: &Vec<Self::Input>, y: &Vec<Self::Output>) {
        assert!(x.len() >= self.k, "Pas assez d’exemples pour k centres");

        // Choix aléatoire initial des centres (k points uniques)
        let mut rng = rand::thread_rng();
        let mut chosen_indices = Vec::new();
        while chosen_indices.len() < self.k {
            let idx = rng.gen_range(0..x.len());
            if !chosen_indices.contains(&idx) {
                chosen_indices.push(idx);
            }
        }
        self.centers = chosen_indices.iter().map(|&i| x[i].clone()).collect();

        // K-means simplifié
        for _ in 0..self.max_kmeans_iter {
            let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.k];

            for xi in x.iter() {
                let (closest_idx, _) = self.centers.iter().enumerate()
                    .map(|(i, c)| (i, squared_euclidean_distance(xi, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                clusters[closest_idx].push(xi.clone());
            }

            let mut new_centers = Vec::new();
            for cluster in clusters.iter() {
                if cluster.is_empty() {
                    new_centers.push(vec![0.0; x[0].len()]);
                } else {
                    new_centers.push(mean_vector(cluster));
                }
            }

            if new_centers == self.centers {
                break;
            }
            self.centers = new_centers;
        }

        // Construire matrice Phi (n_samples x k)
        let n_samples = x.len();
        let mut phi = vec![vec![0.0; self.k]; n_samples];
        for i in 0..n_samples {
            for j in 0..self.k {
                phi[i][j] = self.gaussian_rbf(&x[i], &self.centers[j]);
            }
        }

        // Calcul des poids par pseudo-inverse
        let phi_t = transpose(&phi);
        let phi_t_phi = matmul(&phi_t, &phi);
        let phi_t_phi_inv = inverse(&phi_t_phi).expect("Matrice Phi^T Phi non inversible.");
        let phi_t_y = matvecmul(&phi_t, y);
        self.weights = matvecmul(&phi_t_phi_inv, &phi_t_y);
    }

    fn predict(&self, x: &Self::Input) -> Self::Output {
        self.centers.iter().zip(self.weights.iter())
            .map(|(c, w)| w * self.gaussian_rbf(x, c))
            .sum()
    }
}
