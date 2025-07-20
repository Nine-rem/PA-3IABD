use pyo3::prelude::*;

mod common;
mod linear;
mod perceptron;
mod pmc;
mod rbfn;
mod svm;

/// point d’entrée Python `import ml_library`
#[pymodule]
fn ml_library(py: Python, m: &PyModule) -> PyResult<()> {

    let common_mod = PyModule::new(py, "common")?;

    common::common(py, common_mod)?;
    m.add_submodule(common_mod)?;


    let linear_mod = PyModule::new(py, "linear")?;
    linear::linear(py, linear_mod)?;
    m.add_submodule(linear_mod)?;

    // et de même pour tous vos autres sous-modules…
    let percep_mod = PyModule::new(py, "perceptron")?;
    perceptron::perceptron(py, percep_mod)?;
    m.add_submodule(percep_mod)?;

    let pmc_mod = PyModule::new(py, "pmc")?;
    pmc::pmc(py, pmc_mod)?;
    m.add_submodule(pmc_mod)?;

    let rbfn_mod = PyModule::new(py, "rbfn")?;
    rbfn::rbfn(py, rbfn_mod)?;
    m.add_submodule(rbfn_mod)?;

    let svm_mod = PyModule::new(py, "svm")?;
    svm::svm(py, svm_mod)?;
    m.add_submodule(svm_mod)?;

    Ok(())
}
