use pyo3::prelude::*;

#[pymodule]
pub fn perceptron(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::perceptron::model::Perceptron>()?;
    Ok(())
}
mod model;
mod train;
mod tests;

pub use model::Perceptron;