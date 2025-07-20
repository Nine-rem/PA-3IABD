use pyo3::prelude::*;

#[pymodule]
pub fn svm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::svm::model::SVM>()?;
    Ok(())
}

pub use model::SVM;
mod model;
mod train;
mod tests;
