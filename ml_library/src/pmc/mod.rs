use pyo3::prelude::*;

#[pymodule]
pub fn pmc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::pmc::model::MLP>()?;
    Ok(())
}

pub mod model;
pub mod train;
pub mod tests;
pub use model::MLP;
