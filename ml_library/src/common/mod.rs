use pyo3::prelude::*;

pub mod activations;
pub mod loss;
pub mod dataset;
pub mod math;
pub mod metrics;

/// point d’entrée du sous-module Python `common`
#[pymodule]
pub fn common(py: Python, m: &PyModule) -> PyResult<()> {
    // activations
    let act_mod = PyModule::new(py, "activations")?;
    activations::activations(py, act_mod)?;
    m.add_submodule(act_mod)?;
    // loss
    let loss_mod = PyModule::new(py, "loss")?;
    loss::loss(py, loss_mod)?;
    m.add_submodule(loss_mod)?;
    // dataset
    let data_mod = PyModule::new(py, "dataset")?;
    dataset::dataset(py, data_mod)?;
    m.add_submodule(data_mod)?;
    // math
    let math_mod = PyModule::new(py, "math")?;
    math::math(py, math_mod)?;
    m.add_submodule(math_mod)?;
    // metrics
    let metrics_mod = PyModule::new(py, "metrics")?;
    metrics::metrics(py, metrics_mod)?;
    m.add_submodule(metrics_mod)?;

    Ok(())
}
