pub mod candle_inference;
pub mod candle_qnet;
pub mod env;
pub mod grid;
#[cfg(feature = "python")]
pub mod gumbel;
// pub mod mcts;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use env::{PacmanGym, PacmanGymConfiguration};
// use mcts::MCTSContext;

/// A Python module containing Rust implementations of the PacBot environment.
#[cfg(feature = "python")]
#[pymodule]
fn pacbot_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PacmanGymConfiguration>()?;
    m.add_class::<PacmanGym>()?;
    m.add_class::<gumbel::GumbelConfig>()?;
    m.add_function(wrap_pyfunction!(gumbel::gumbel_search, m)?)?;
    // m.add_class::<MCTSContext>()?;
    Ok(())
}
