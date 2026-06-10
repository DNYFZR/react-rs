# Package API Configuration
# NOTE: IDE linting will show an error for .react_rs imports (ignore them)
import typing as _typing

import polars as _pl

# Import native Python code
from ._weibull import Weibull

# Define functionality to be made available to users
__all__ = [
    # Rust modules
    "simulate",
    "constrain",
    "aggregate",
    "profile",
    # Native Python modules
    "Weibull",
]


def _py_data_frame(rust_data_frame) -> _pl.DataFrame:
    """
    Internal method for converting Rust DataFrame objects to Python DataFrames

    DO NOT USE IN PYTHON
    - FOR USE ON RUST BACKEND OUTPUTS ONLY

    Inputs
    ---
    - rust_frame : polars DataFrame (Rust only)

    Returns
    ---
    polars DataFrame
    """
    return _pl.DataFrame(rust_data_frame)


# Implement Rust backend functionality in Python
def simulate(
    df: _pl.DataFrame,
    id_col: str,
    age_col: str,
    cost_col: str,
    probabilities: _typing.List[float],
    n_sims: int,
    n_steps: int,
    parallel_limit: int,
) -> _pl.DataFrame:
    """
    Simulate (Rust)
    ---

    Execute a discrete event simulation via Rust using a Weibull model

    Inputs
    ---
    - df : polars DataFrame containing input information for simulation items
    - id_col : string name of column in df containing items unique ID
    - age_col : string name of column in df containing initial age of each item
    - cost_col : string name of column in df containing value of each item
    - probabilities : list of floats represending the survival curve of the items
    - n_sims : int value for the number of simulations to execute on the complete set of items
    - n_steps : int value for the number of timesteps to execute the simulation over
    - para_limit : int value for the maximum number of parallelised simulations to run at any one time

    Returns
    ---
    polars.DataFrame
    """
    from .react_rs import simulate as rs_sim

    return _py_data_frame(
        rs_sim(
            df=df,
            id_col=id_col,
            age_col=age_col,
            cost_col=cost_col,
            probabilities=probabilities,
            n_sims=n_sims,
            n_steps=n_steps,
            para_limit=parallel_limit,
        )
    )


def constrain(
    df: _pl.DataFrame,
    constrain_steps: int,
    iter_regex: str,
    cost_col: str,
    constraints: _typing.List[int],
    partition_by: str,
    parallel_limit: int,
) -> _pl.DataFrame:
    """
    Constrain (Rust)
    ---

    Apply a financial limit to the output of a simulation & reallocate assets
    within each unique simulation output

    Inputs
    ---
    - df : polars DataFrame containing the output of a simulation to be constrained
    - iter_regex : string pattern for accessing the unique timesteps in the
    - cost_col : string name of column in df containing value of each item
    - constraints : list of integers representing the financial limit to be applied
    in each timestep (length must match number of timesteps in simulation output)
    - partition_by : string column name containing the simulation ID
    - run_method : string trigger for rust run method - options: full / batched / parallel

    Returns
    ---
    polars.DataFrame
    """
    from .react_rs import constrain as rs_constrain

    return _py_data_frame(
        rs_constrain(
            df=df,
            constrain_steps=constrain_steps,
            iter_regex=iter_regex,
            cost_col=cost_col,
            constraints=constraints,
            partition_by=partition_by,
            para_limit=parallel_limit,
        )
    )


def aggregate(
    df: _pl.DataFrame,
    partition_by: str,
    iter_regex: str,
    target_value: int,
    cost_col: str | None,
) -> _pl.DataFrame:
    """
    Aggregate (Rust)
    ---

    Inputs
    ---
    - df : polars DataFrame containing the aggregation target table
    - partition_by : string column name containing the simulation ID
    - iter_regex : string pattern for accessing the unique timesteps in the
    simulation output
    - target_value : int value to target for aggregations, typically this will be 0
    to represent when an asset is replaced
    - cost_col : optional
        - string column name containing the value of the items
        - if provided, target_value occurances will be converted from count to value

    Returns
    ---
    polars.DataFrame
    """
    from .react_rs import aggregate as rs_agg

    return _py_data_frame(
        rs_agg(
            df=df,
            partition_by=partition_by,
            iter_regex=iter_regex,
            target_value=target_value,
            cost_col=cost_col,
        )
    )


def profile(
    df: _pl.DataFrame,
    partition_by: str,
    iter_regex: str,
) -> _pl.DataFrame:
    """
    Profile (Rust)
    ---

    Create a value count for a DataFrame located @ filepath

    Inputs
    ---
    - df : polars DataFrame containing the profile target table
    - partition_by : string column name containing the simulation ID
    - iter_regex : string pattern for accessing the unique timesteps in the
    simulation output

    Returns
    ---
    polars DataFrame

    """
    from .react_rs import profile as rs_profile

    return _py_data_frame(
        rs_profile(
            df=df,
            partition_by=partition_by,
            iter_regex=iter_regex,
        )
    )
