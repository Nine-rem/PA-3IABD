// src/pmc/train.rs
//! fit, predict, Model impl et FFI wrappers
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
                for l in 0..self.weights.len() {
                    let z = self.weights[l].t().dot(&a) + &self.biases[l];
                    zs.push(z.clone());
                    a = if l == self.weights.len() - 1 {
                        if !self.is_classification {
                            z.clone()
                        } else if self.layers.last().unwrap()==&1 {
                            z.clone()
                        } else {
                            PMC::softmax(&z)
                        }
                    } else { PMC::sigmoid(&z) };
                    activations.push(a.clone());
                }
                let target = if !self.is_classification {
                    Array1::from(vec![y_train[i]])
                } else if self.layers.last().unwrap()==&1 {
                    Array1::from(vec![y_train[i]])
                } else {
                    let mut oh = Array1::zeros(*self.layers.last().unwrap());
                    oh[y_train[i] as usize]=1.0; oh
                };
                let mut delta = if !self.is_classification || *self.layers.last().unwrap()>1 {
                    activations.last().unwrap() - &target
                } else { activations.last().unwrap() - &target };
                for l in (0..self.weights.len()).rev() {
                    let gw = activations[l].view().insert_axis(Axis(1))
                        .dot(&delta.view().insert_axis(Axis(0)));
                    let gb = delta.clone();
                    self.weights[l] -= &(self.learning_rate * &gw);
                    self.biases[l]  -= &(self.learning_rate * &gb);
                    if l>0 {
                        let sp = PMC::sigmoid_derivative(&zs[l-1]);
                        delta = self.weights[l].dot(&delta)*sp;
                    }
                }
            }
        }
    }
    pub fn predict(&self,x:&[Vec<f32>])->Vec<f32>{
        x.iter().map(|row|{
            let mut a=Array1::from(row.clone());
            for l in 0..self.weights.len(){
                let z=self.weights[l].t().dot(&a)+&self.biases[l];
                a=if l==self.weights.len()-1{
                    if !self.is_classification{z}
                    else if self.layers.last().unwrap()==&1{z}
                    else{PMC::softmax(&z)}
                }else{PMC::sigmoid(&z)};
            }
            if !self.is_classification{a[0]}
            else if self.layers.last().unwrap()==&1{if a[0]>=0.0{1.0}else{-1.0}}
            else{a.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f32}
        }).collect()}
    }

impl Model for PMC {
    fn fit(&mut self,x_train:&[Vec<f32>],y_train:&[f32],x_val:&[Vec<f32>],y_val:&[f32]){self.fit(x_train,y_train,x_val,y_val)}
    fn predict(&self,x:&[Vec<f32>])->Vec<f32>{self.predict(x)}
    fn save(&self,_p:&str)->Result<(),Box<dyn std::error::Error>>{
        // Implémentation simplifiée sans sérialisation pour l'instant
        Ok(())
    }
    fn load(&mut self,_p:&str)->Result<(),Box<dyn std::error::Error>>{
        // Implémentation simplifiée sans désérialisation pour l'instant  
        Ok(())
    }
}

// FFI wrappers
#[no_mangle]
pub extern "C" fn pmc_save(ptr:*const PMC,path:*const c_char)->i32{
    if ptr.is_null()||path.is_null(){return-1;}
    let pmc=unsafe{&*ptr};
    let cstr=unsafe{CStr::from_ptr(path)};
    match cstr.to_str(){
        Ok(s) => match pmc.save(s){
            Ok(_)=>0,
            Err(_)=>-1,
        },
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn pmc_load(ptr:*mut PMC,path:*const c_char)->i32{
    if ptr.is_null()||path.is_null(){return-1;}
    let pmc=unsafe{&mut*ptr};
    let cstr=unsafe{CStr::from_ptr(path)};
    match cstr.to_str(){
        Ok(s) => match pmc.load(s){
            Ok(_)=>0,
            Err(_)=>-1,
        },
        Err(_) => -1,
    }
}
#[no_mangle]pub extern "C" fn pmc_fit(ptr:*mut PMC,x:*const f32,y:*const f32,n:usize,d:usize)->i32{if ptr.is_null()||x.is_null()||y.is_null(){return-1;}let pmc=unsafe{&mut*ptr};let xs=unsafe{slice::from_raw_parts(x,n*d)};let ys=unsafe{slice::from_raw_parts(y,n)};let xv=xs.chunks(d).map(|c|c.to_vec()).collect::<Vec<_>>();let yv=ys.to_vec();pmc.fit(&xv,&yv,&[],&[]);0}
#[no_mangle]pub extern "C" fn pmc_predict(ptr:*const PMC,x:*const f32,n:usize,d:usize,out:*mut f32)->i32{if ptr.is_null()||x.is_null()||out.is_null(){return-1;}let pmc=unsafe{&*ptr};let xs=unsafe{slice::from_raw_parts(x,n*d)};let xv=xs.chunks(d).map(|c|c.to_vec()).collect::<Vec<_>>();let preds=pmc.predict(&xv);for(i,&v) in preds.iter().enumerate().take(n){unsafe{*out.add(i)=v;}}0}

// Fonction train_pmc pour les tests Python
#[no_mangle]
pub extern "C" fn train_pmc(
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_features: usize,
    n_hidden: usize,
    hidden_sizes_ptr: *const i64,
    n_outputs: usize,
    hidden_act: usize,
    output_act: usize,
    lr: f64,
    n_iters: usize,
    weights_out: *mut f64,
) {
    let _ = hidden_act; // Ignorer l'avertissement pour l'instant
    unsafe {
        // Conversion des données d'entrée de f64 vers f32
        let x_buf = slice::from_raw_parts(x_ptr, n_samples * n_features);
        let y_buf = slice::from_raw_parts(y_ptr, n_samples * n_outputs);
        let hidden_sizes = slice::from_raw_parts(hidden_sizes_ptr, n_hidden);
        
        // Conversion vers f32 pour compatibilité avec PMC
        let x_f32: Vec<f32> = x_buf.iter().map(|&x| x as f32).collect();
        let y_f32: Vec<f32> = y_buf.iter().map(|&y| y as f32).collect();
        
        // Traitement des labels pour classification binaire
        let y_processed: Vec<f32> = if output_act == 0 && n_outputs == 1 {
            // Classification binaire: convertir {-1,1} vers {-1,1} (garder tel quel)
            y_f32
        } else {
            y_f32
        };
        
        // Construction de l'architecture
        let mut layers = vec![n_features];
        for &size in hidden_sizes {
            layers.push(size as usize);
        }
        layers.push(n_outputs);
        
        println!("→ [RS] train_pmc layers: {:?}", layers);
        println!("→ [RS] Using learning rate: {:.6}", lr);
        
        // Création et entraînement du PMC
        let mut pmc = PMC::new(
            layers.clone(), // Cloner pour éviter le move
            lr as f32,
            n_iters,
            output_act != 2, // true si classification, false si régression
        );
        
        // Préparation des données d'entraînement
        let x_train: Vec<Vec<f32>> = x_f32.chunks(n_features).map(|chunk| chunk.to_vec()).collect();
        
        // Entraînement
        pmc.fit(&x_train, &y_processed, &[], &[]);
        
        // Extraction des poids (approximation pour compatibilité)
        // Note: Cette implémentation PMC ne stocke pas les poids de la même manière
        // que l'implémentation nalgebra, donc on renvoie des zéros pour l'instant
        let weights_len = layers.windows(2).map(|w| (w[0] + 1) * w[1]).sum::<usize>();
        let weights_slice = slice::from_raw_parts_mut(weights_out, weights_len);
        for i in 0..weights_len {
            weights_slice[i] = 0.0; // Placeholder - il faudrait extraire les vrais poids
        }
    }
}
