use pyo3::prelude::*;

#[pymodule]
pub fn linear(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::linear::model::LinearModel>()?;
    Ok(())
}

pub use model::LinearModel;
mod model;
mod train;
#[cfg(test)]
mod tests;
