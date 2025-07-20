use pyo3::prelude::*;

#[pymodule]
pub fn rbfn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::rbfn::model::RBFN>()?;
    Ok(())
}

pub use model::RBFN;
mod model;
mod train;
mod tests;
