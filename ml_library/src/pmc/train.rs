use crate::common::math::dot_product;
use crate::pmc::model::MLP;

fn backward_propagation(network: &mut MLP, expected: &[f64]) {
    let last = network.layers.len() - 1;
    let out_layer = &mut network.layers[last];
    for j in 0..out_layer.outputs.len() {
        let err = expected[j] - out_layer.outputs[j];
        out_layer.deltas[j] = err * (1.0 - out_layer.outputs[j].powi(2));
    }

    for l in (0..last).rev() {
        let (left, right) = network.layers.split_at_mut(l + 1);
        let layer = &mut left[l];
        let next = &right[0];
        for i in 0..layer.outputs.len() {
            let err = dot_product(&next.weights[i], &next.deltas);
            layer.deltas[i] = err * (1.0 - layer.outputs[i].powi(2));
        }
    }
}

fn update_weights(network: &mut MLP, input: &[f64]) {
    let mut activations = input.to_vec();
    for layer in &mut network.layers {
        for i in 0..activations.len() {
            for j in 0..layer.outputs.len() {
                layer.weights[i][j] += network.learning_rate * layer.deltas[j] * activations[i];
            }
        }
        let bias_row = layer.weights.len() - 1;
        for j in 0..layer.outputs.len() {
            layer.weights[bias_row][j] += network.learning_rate * layer.deltas[j];
        }
        activations = layer.outputs.clone();
    }
}

pub fn train_mlp(
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    hidden_sizes: Vec<usize>,
    n_outputs: usize,
    iterations: usize,
    learning_rate: f64,
) -> MLP {
    assert_eq!(inputs.len(), targets.len(), "inputs.len() must equal targets.len()");
    let mut net = MLP::new_py(inputs[0].len(), hidden_sizes, n_outputs, learning_rate);
    for _ in 0..iterations {
        for (x, y) in inputs.iter().zip(targets.iter()) {
            net.predict_internal(x);
            backward_propagation(&mut net, y);
            update_weights(&mut net, x);
        }
    }
    net
}
