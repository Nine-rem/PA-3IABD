
use ndarray::{Array1, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;
use crate::{Model, pmc::model::PMC};

impl PMC {
    pub fn fit(&mut self,
        x_train: &[Vec<f32>], y_train: &[f32], _x_val: &[Vec<f32>], _y_val: &[f32],
    ) {
        let n = x_train.len();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = thread_rng();
        
        for _ in 0..self.epochs {
            idx.shuffle(&mut rng);
            for &i in &idx {
                let mut a = Array1::from(x_train[i].clone());
                let mut activations = vec![a.clone()];
                let mut zs = Vec::with_capacity(self.layers.len() - 1);
                
                // Forward pass
                for l in 0..self.weights.len() {
                    let z = self.weights[l].t().dot(&a) + &self.biases[l];
                    zs.push(z.clone());
                    a = if l == self.weights.len() - 1 {
                        if !self.is_classification {
                            z.clone()
                        } else if self.layers.last().unwrap() == &1 {
                            PMC::sigmoid(&z)
                        } else {
                            PMC::softmax(&z)
                        }
                    } else { 
                        PMC::sigmoid(&z) 
                    };
                    activations.push(a.clone());
                }
                
                let target = if !self.is_classification {
                    Array1::from(vec![y_train[i]])
                } else if self.layers.last().unwrap() == &1 {
                    Array1::from(vec![y_train[i]])
                } else {
                    let mut oh = Array1::zeros(*self.layers.last().unwrap());
                    oh[y_train[i] as usize] = 1.0; 
                    oh
                };
                
                // Backward pass avec normalisation des gradients
                let mut delta = activations.last().unwrap() - &target;
                
                for l in (0..self.weights.len()).rev() {
                    let gw = activations[l].view().insert_axis(Axis(1))
                        .dot(&delta.view().insert_axis(Axis(0)));
                    let gb = delta.clone();
                    
                    // CORRECTION: Normaliser les gradients par n pour éviter la divergence
                    let normalized_lr = self.learning_rate / (n as f32);
                    self.weights[l] -= &(normalized_lr * &gw);
                    self.biases[l] -= &(normalized_lr * &gb);
                    
                    if l > 0 {
                        let sp = PMC::sigmoid_derivative(&activations[l]);
                        delta = self.weights[l].dot(&delta) * sp;
                    }
                }
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter().map(|row| {
            let mut a = Array1::from(row.clone());
            for l in 0..self.weights.len() {
                let z = self.weights[l].t().dot(&a) + &self.biases[l];
                a = if l == self.weights.len() - 1 {
                    // Couche de sortie
                    if !self.is_classification {
                        z
                    } else if self.layers.last().unwrap() == &1 {
                        PMC::sigmoid(&z)
                    } else {
                        PMC::softmax(&z)
                    }
                } else {
                    // Couches cachées
                    PMC::sigmoid(&z)
                };
            }
            
            if !self.is_classification {
                a[0]
            } else if self.layers.last().unwrap() == &1 {
                if a[0] >= 0.5 { 1.0 } else { -1.0 }
            } else {
                a.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f32
            }
        }).collect()
    }
}

impl Model for PMC {
    fn fit(&mut self, x_train: &[Vec<f32>], y_train: &[f32], x_val: &[Vec<f32>], y_val: &[f32]) {
        self.fit(x_train, y_train, x_val, y_val)
    }
    
    fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        self.predict(x)
    }
    
    fn save(&self, _p: &str) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    
    fn load(&mut self, _p: &str) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

// FFI wrappers
#[no_mangle]
pub extern "C" fn pmc_save(ptr: *const PMC, path: *const c_char) -> i32 {
    if ptr.is_null() || path.is_null() { return -1; }
    let pmc = unsafe { &*ptr };
    let cstr = unsafe { CStr::from_ptr(path) };
    match cstr.to_str() {
        Ok(s) => match pmc.save(s) {
            Ok(_) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn pmc_load(ptr: *mut PMC, path: *const c_char) -> i32 {
    if ptr.is_null() || path.is_null() { return -1; }
    let pmc = unsafe { &mut *ptr };
    let cstr = unsafe { CStr::from_ptr(path) };
    match cstr.to_str() {
        Ok(s) => match pmc.load(s) {
            Ok(_) => 0,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn pmc_fit(ptr: *mut PMC, x: *const f32, y: *const f32, n: usize, d: usize) -> i32 {
    if ptr.is_null() || x.is_null() || y.is_null() { return -1; }
    let pmc = unsafe { &mut *ptr };
    let xs = unsafe { slice::from_raw_parts(x, n * d) };
    let ys = unsafe { slice::from_raw_parts(y, n) };
    let xv = xs.chunks(d).map(|c| c.to_vec()).collect::<Vec<_>>();
    let yv = ys.to_vec();
    pmc.fit(&xv, &yv, &[], &[]);
    0
}

#[no_mangle]
pub extern "C" fn pmc_predict(ptr: *const PMC, x: *const f32, n: usize, d: usize, out: *mut f32) -> i32 {
    if ptr.is_null() || x.is_null() || out.is_null() { return -1; }
    let pmc = unsafe { &*ptr };
    let xs = unsafe { slice::from_raw_parts(x, n * d) };
    let xv = xs.chunks(d).map(|c| c.to_vec()).collect::<Vec<_>>();
    let preds = pmc.predict(&xv);
    for (i, &v) in preds.iter().enumerate().take(n) {
        unsafe { *out.add(i) = v; }
    }
    0
}
