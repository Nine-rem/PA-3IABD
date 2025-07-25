

pub mod common;
pub mod linear;
pub mod perceptron;
pub mod pmc;
pub mod rbfn;
pub mod svm;

// EXPORT FFI - Fonctions nécessaires pour les tests Python
use std::slice;
use rand::seq::SliceRandom;
use rand::thread_rng;

// Fonction d'entraînement basée sur le code qui marche dans pmc/train.rs
fn train_pmc_direct(pmc: &mut pmc::model::PMC, x_train: &[Vec<f32>], y_train: &[f32]) {
    let n = x_train.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = thread_rng();
    
    for _ in 0..pmc.epochs {
        idx.shuffle(&mut rng);
        for &i in &idx {
            use ndarray::{Array1, Axis};
            
            let mut a = Array1::from(x_train[i].clone());
            let mut activations = vec![a.clone()];
            let mut zs = Vec::with_capacity(pmc.layers.len() - 1);
            
            // Forward pass
            for l in 0..pmc.weights.len() {
                let z = pmc.weights[l].t().dot(&a) + &pmc.biases[l];
                zs.push(z.clone());
                a = if l == pmc.weights.len() - 1 {
                    if !pmc.is_classification {
                        z.clone()
                    } else if pmc.layers.last().unwrap() == &1 {
                        pmc::model::PMC::sigmoid(&z)
                    } else {
                        pmc::model::PMC::softmax(&z)
                    }
                } else { 
                    pmc::model::PMC::sigmoid(&z) 
                };
                activations.push(a.clone());
            }
            
            let target = if !pmc.is_classification {
                Array1::from(vec![y_train[i]])
            } else if pmc.layers.last().unwrap() == &1 {
                Array1::from(vec![y_train[i]])
            } else {
                let mut oh = Array1::zeros(*pmc.layers.last().unwrap());
                oh[y_train[i] as usize] = 1.0; 
                oh
            };
            
            // Backward pass avec normalisation par le nombre d'échantillons
            let mut delta = activations.last().unwrap() - &target;
            
            for l in (0..pmc.weights.len()).rev() {
                let gw = activations[l].view().insert_axis(Axis(1))
                    .dot(&delta.view().insert_axis(Axis(0)));
                let gb = delta.clone();
                
                // CORRECTION: Normaliser les gradients par n pour éviter la divergence
                let normalized_lr = pmc.learning_rate / (n as f32);
                pmc.weights[l] -= &(normalized_lr * &gw);
                pmc.biases[l] -= &(normalized_lr * &gb);
                
                if l > 0 {
                    let sp = pmc::model::PMC::sigmoid_derivative(&activations[l]);
                    delta = pmc.weights[l].dot(&delta) * sp;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn train_pmc(
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    n_hidden: usize,
    hidden_sizes_ptr: *const i64,
    n_outputs: usize,
    _hidden_act: usize, // Ignoré pour simplifier
    output_act: usize,
    lr: f64,
    n_iters: usize,
    weights_out: *mut f64,
) {
    unsafe {
        if x_ptr.is_null() || y_ptr.is_null() || hidden_sizes_ptr.is_null() || weights_out.is_null() {
            return;
        }

        let x_buf = slice::from_raw_parts(x_ptr, n_samples * n_features);
        let y_buf = slice::from_raw_parts(y_ptr, n_samples * n_outputs);
        let hidden_sizes = slice::from_raw_parts(hidden_sizes_ptr, n_hidden);
        
        let x_f32: Vec<f32> = x_buf.iter().map(|&x| x as f32).collect();
        let y_f32: Vec<f32> = y_buf.iter().map(|&y| y as f32).collect();
        
        // Architecture
        let mut layers = vec![n_features];
        for &size in hidden_sizes {
            layers.push(size as usize);
        }
        layers.push(n_outputs);
        
        // PMC
        let mut pmc = pmc::model::PMC::new(
            layers.clone(),
            lr as f32,
            n_iters,
            output_act != 2,
        );
        
        let x_train: Vec<Vec<f32>> = x_f32.chunks(n_features).map(|chunk| chunk.to_vec()).collect();
        
        // Utiliser l'implémentation directe d'entraînement SGD efficace
        train_pmc_direct(&mut pmc, &x_train, &y_f32);
        
        // Export poids
        let weights_len = layers.windows(2).map(|w| (w[0] + 1) * w[1]).sum::<usize>();
        let weights_slice = slice::from_raw_parts_mut(weights_out, weights_len);
        
        let mut idx = 0;
        for (layer_idx, weight_matrix) in pmc.weights.iter().enumerate() {
            let (rows, cols) = weight_matrix.dim();
            for col in 0..cols {
                for row in 0..rows {
                    weights_slice[idx] = weight_matrix[[row, col]] as f64;
                    idx += 1;
                }
                weights_slice[idx] = pmc.biases[layer_idx][col] as f64;
                idx += 1;
            }
        }
    }
}

/// Trait générique pour tous les modèles
pub trait Model {
    /// Entraînement avec données d'entraînement et validation
    fn fit(
        &mut self,
        x_train: &[Vec<f32>],
        y_train: &[f32],
        x_val: &[Vec<f32>],
        y_val: &[f32],
    );

    /// Prédiction: renvoie un vecteur de sorties
    fn predict(&self, x: &[Vec<f32>]) -> Vec<f32>;

    /// Sauvegarde dans un fichier (binaire ou texte)
    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;

    /// Chargement depuis un fichier
    fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}