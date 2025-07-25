/*
use crate::common::model::{Matrix, ModelError, Vector};
use crate::linear::LinearModel;
use crate::pmc::PMC;
use rand::{SeedableRng, seq::SliceRandom};
use rand::rngs::StdRng;

/// Entraînement pour LinearModel
pub fn train_linear(
    model: &mut LinearModel,
    x_train: &Matrix,
    y_train: &Vector,
    x_val: &Matrix,
    y_val: &Vector,
) -> Result<(), ModelError> {
    let n = x_train.nrows();
    let d = x_train.ncols();
    // construit X avec biais
    let mut Xmat = Matrix::zeros(n, d + 1);
    for i in 0..n {
        Xmat[(i, 0)] = 1.0;
        for j in 0..d {
            Xmat[(i, j + 1)] = x_train[(i, j)];
        }
    }
    // détermine nb de classes
    let y_max = *y_train.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_min = *y_train.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let multiclass = model.is_classification && !(y_max == 1.0 && (y_min == 0.0 || y_min == -1.0));
    let num_classes = if model.is_classification {
        if multiclass { y_max as usize + 1 } else { 1 }
    } else { 1 };

    model.weights = Matrix::zeros(d + 1, num_classes);
    model.train_loss_history.clear();
    model.val_loss_history.clear();
    model.train_accuracy_history.clear();
    model.val_accuracy_history.clear();

    let mut rng = StdRng::seed_from_u64(42);
    let mut idx: Vec<usize> = (0..n).collect();

    for _ in 0..model.epochs {
        idx.shuffle(&mut rng);
        let mut epoch_loss = 0.0;

        for &i in &idx {
            let xi = Xmat.row(i).transpose();
            let yi = if !model.is_classification {
                Vector::from_element(1, y_train[i])
            } else if num_classes == 1 {
                let mut lab = y_train[i];
                if !model.binary_labels_are_minus1_plus1 {
                    lab = if lab == 0.0 { -1.0 } else { 1.0 };
                }
                Vector::from_element(1, lab)
            } else {
                let mut v = Vector::zeros(num_classes);
                v[y_train[i] as usize] = 1.0;
                v
            };

            let pred = &model.weights.transpose() * &xi;
            let err = &pred - yi.clone();
            model.weights -= model.learning_rate * (&xi * err.transpose());
            epoch_loss += err.norm_squared();
        }

        model.train_loss_history.push(epoch_loss / (n as f64));

        // accuracy
        let train_acc = model.predict(x_train)?
            .iter().zip(y_train.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / (n as f64);
        let val_acc = model.predict(x_val)?
            .iter().zip(y_val.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / (x_val.nrows() as f64);

        model.train_accuracy_history.push(train_acc);
        model.val_accuracy_history.push(val_acc);

        // loss validation
        let mut val_loss = 0.0;
        for i in 0..x_val.nrows() {
            let xi = x_val.row(i).transpose();
            let yi = if !model.is_classification {
                Vector::from_element(1, y_val[i])
            } else if num_classes == 1 {
                let mut lab = y_val[i];
                if !model.binary_labels_are_minus1_plus1 {
                    lab = if lab == 0.0 { -1.0 } else { 1.0 };
                }
                Vector::from_element(1, lab)
            } else {
                let mut v = Vector::zeros(num_classes);
                v[y_val[i] as usize] = 1.0;
                v
            };
            let pred = &model.weights.transpose() * &xi;
            val_loss += (&pred - yi).norm_squared();
        }
        model.val_loss_history.push(val_loss / (x_val.nrows() as f64));
    }

    Ok(())
}

/// Entraînement pour PMC
pub fn train_pmc(
    model: &mut PMC,
    x_train: &Matrix,
    y_train: &Vector,
    x_val: &Matrix,
    y_val: &Vector,
) -> Result<(), ModelError> {
    model.train_loss_history.clear();
    model.val_loss_history.clear();
    model.train_accuracy_history.clear();
    model.val_accuracy_history.clear();
    let mut rng = StdRng::seed_from_u64(42);
    let n = x_train.nrows();
    let mut idx: Vec<usize> = (0..n).collect();

    for _ in 0..model.epochs {
        idx.shuffle(&mut rng);
        for &i in &idx {
            let input = x_train.row(i).transpose();
            let mut acts = Vec::new();
            let mut zs = Vec::new();
            let out = model.forward(&input, &mut acts, &mut zs);

            let target = if !model.is_classification {
                Vector::from_element(1, y_train[i])
            } else if *model.layers.last().unwrap() == 1 {
                let mut lab = y_train[i];
                lab = if lab == 0.0 { -1.0 } else { 1.0 };
                Vector::from_element(1, lab)
            } else {
                let mut v = Vector::zeros(*model.layers.last().unwrap());
                v[y_train[i] as usize] = 1.0;
                v
            };

            // backprop…
            let mut delta = vec![Vector::zeros(0); model.weights.len()];
            let last = &zs[zs.len() - 1];
            if !model.is_classification {
                delta[delta.len() - 1] = 2.0 * (&out - &target);
            } else {
                let deriv = last.map(|x| PMC::sigmoid_derivative(x));
                delta[delta.len() - 1] = (2.0 * (&out - &target)).component_mul(&deriv);
            }
            for l in (0..model.weights.len() - 1).rev() {
                let sp = zs[l].map(|x| PMC::sigmoid_derivative(x));
                delta[l] = model.weights[l + 1].transpose() * &delta[l + 1];
                delta[l] = delta[l].component_mul(&sp);
            }
            for l in 0..model.weights.len() {
                model.weights[l] -= model.learning_rate * (&delta[l] * acts[l].transpose());
                model.biases[l] -= model.learning_rate * &delta[l];
            }
        }

        model.train_loss_history.push(model.compute_loss_total(x_train, y_train));
        model.val_loss_history.push(model.compute_loss_total(x_val, y_val));
        model.train_accuracy_history.push(model.compute_accuracy(x_train, y_train));
        model.val_accuracy_history.push(model.compute_accuracy(x_val, y_val));
    }

    Ok(())
}
*/