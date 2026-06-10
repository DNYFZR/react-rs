// Discrete Event Model
// Built with Rust & Compiled into a Python Library via PyO3 & Maturin
mod agg;
mod io;
mod sim;
mod tx;

use pyo3::prelude::*;

#[pymodule]
mod react_rs {
    use crate::agg;
    use crate::sim;
    use crate::tx;

    use polars::error::PolarsError;
    use polars::frame::DataFrame;
    use pyo3::IntoPyObjectExt;
    use pyo3::prelude::*;
    use pyo3_polars::PyDataFrame;

    #[pymodule]
    fn pyfunctions(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Method for exposing functions to Python runtime
        m.add_function(wrap_pyfunction!(simulate, m)?)?;
        m.add_function(wrap_pyfunction!(constrain, m)?)?;
        m.add_function(wrap_pyfunction!(aggregate, m)?)?;
        m.add_function(wrap_pyfunction!(profile, m)?)?;
        Ok(())
    }

    // Internal package functions for Python <-> Rust DataFrame handling
    fn import_py_dataframe(py_df: &Bound<'_, PyAny>) -> Result<DataFrame, PolarsError> {
        match py_df.extract::<PyDataFrame>() {
            Ok(df) => Ok(df.into()),
            Err(e) => Err(e.into()),
        }
    }

    fn return_py_dataframe(rust_output: Result<DataFrame, PolarsError>) -> PyResult<Py<PyAny>> {
        // Internal function to handle conversion of Rust output to Python
        match rust_output {
            Ok(frame) => Python::attach(|py| {
                // let py_df: Py<PyAny> = (PyDataFrame { df: frame.into() }).into_py_any(py).unwrap();
                let py_df: Py<PyAny> = (PyDataFrame(frame.into())).into_py_any(py).unwrap();
                Ok(py_df)
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    fn simulate(
        df: &Bound<'_, PyAny>,
        id_col: &str,
        age_col: &str,
        cost_col: &str,
        probabilities: Vec<f64>,
        n_sims: i64,
        n_steps: i64,
        para_limit: i64,
    ) -> PyResult<Py<PyAny>> {
        // Convert Python dataset to Rust
        let df = match import_py_dataframe(df) {
            Ok(df) => df,
            Err(e) => panic!("{}", e.to_string()),
        };
        let uuids = tx::col_to_vec_str(&df, id_col);
        let states = tx::col_to_vec_i64(&df, age_col);
        let costs = tx::col_to_vec_i64(&df, cost_col);

        // Execute discrete event simulation
        return return_py_dataframe(sim::engine(
            n_sims,
            n_steps,
            uuids,
            states,
            probabilities,
            Some(&costs),
            para_limit,
        ));
    }

    #[pyfunction]
    fn constrain(
        df: &Bound<'_, PyAny>,
        constrain_steps: i64,
        iter_regex: &str,
        cost_col: &str,
        constraints: Vec<i64>,
        partition_by: &str,
        para_limit: i64,
    ) -> PyResult<Py<PyAny>> {
        // Convert Python dataset to Rust
        let df = match import_py_dataframe(df) {
            Ok(df) => df,
            Err(e) => panic!("{}", e.to_string()),
        };

        // Apply constraints to simulation output
        return return_py_dataframe(tx::constrain_parallel(
            &df,
            constrain_steps,
            iter_regex,
            cost_col,
            constraints,
            partition_by,
            para_limit,
        ));
    }

    #[pyfunction]
    fn aggregate(
        df: &Bound<'_, PyAny>,
        partition_by: &str,
        iter_regex: &str,
        target_value: i64,
        cost_col: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        // Convert Python dataset to Rust
        let df = match import_py_dataframe(df) {
            Ok(df) => df,
            Err(e) => panic!("{}", e.to_string()),
        };

        // Execute aggregations
        return return_py_dataframe(agg::aggregate(
            df,
            partition_by,
            iter_regex,
            target_value,
            cost_col,
        ));
    }

    #[pyfunction]
    fn profile(df: &Bound<'_, PyAny>, partition_by: &str, iter_regex: &str) -> PyResult<Py<PyAny>> {
        // Convert Python dataset to Rust
        let df = match import_py_dataframe(df) {
            Ok(df) => df,
            Err(e) => panic!("{}", e.to_string()),
        };

        // Execute value count on dataframe
        return return_py_dataframe(agg::count_values(&df, partition_by, iter_regex));
    }
}
