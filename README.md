<h1 align="center"><b> REACT_RS </b></h1>

## **Summary**

React_rs (reactors) is Python package with Rust backend functionality for executing Discrete Event simulations in a fast & cost efficient way. 

- Exposed Rust functions have a corresponding Python function in the [API definition](python/react_rs/__init__.py) which allows the application to convert the Rust polars DataFrame into a Python polars DataFrame & vice versa.

- There are also native Python components to the package, which are in dedicated scripts in the [Python directory](python/react_rs).

The core functionality of the app is : 

- Fit, adjust & generate of a range of Weibull model survival curves

- Simulation of discrete events over a user-defined number of timesteps & simulations based on the survival curve in the probabilities column of the input dataset - these can be from the internal Weibull module, or from any other user-defined model

- Application of financial limits to simulation outputs, and reallocation of replacement events in line with these limits

- Event or cost based aggregations are summaries of the totals for each simulation & timestep

- Profile based aggregation creates a summary of the count of items ages at each timestep within each simulation iteration

## **Usage**

```py
import polars as pl
import react_rs

# Setup survival curve config
model_config = {
    "general": {"states": [15, 25], "values": [0.95, 0.4]},
    "short": {"base_model": "general", "mean_age": 15},
    "medium": {"base_model": "general", "mean_age": 25},
    "long": {"base_model": "general", "mean_age": 35},
}

# Generate survival curves via Weibull module
wb = react_rs.Weibull()

for model, params in model_config.items():
    # Fit method auto-handles base model & adjustments to mean
    wb.fit(model_name=model, **params)

# Generate survival curve DataFrame under 'curves' attribute
wb.generate()

# Join survival curves DataFrame to input DataFrame
df = pl.read_parquet("./tests/data/input.parquet").join(
    pl.DataFrame({"model": wb.curves.keys(), "curve": wb.curves.values()}),
    "model",
    "left",
)

# Execute simulation in Rust
sim_result = react_rs.simulate(
    df=df,
    id_col="uuid",
    age_col="step_0",
    cost_col="value",
    probs_col="curve",
    n_sims=100,
    n_steps=50,
    parallel_limit=10,
)

# Constrain simulation output in Rust
sim_result_constrained = react_rs.constrain(
    df=sim_result,
    constrain_steps=30, # limit steps taken through constraint system
    iter_regex="step",
    cost_col="cost",
    constraints=[int(50e6) for _ in range(30)],
    partition_by="sim_id",
    parallel_limit=10,
)

# Aggregate simulation outputs
sim_result_agg = react_rs.aggregate(
    df=sim_result,
    partition_by="sim_id",
    iter_regex="step",
    target_value=0,
    cost_col="cost", # None if events are req'd
)

sim_result_const_agg = react_rs.aggregate(
    df=sim_result_constrained,
    partition_by="sim_id",
    iter_regex="step",
    target_value=0,
    cost_col="cost",
)

# Age profile across iterations & timesteps
sim_profile = react_rs.profile(
    df=sim_result,
    partition_by="sim_id",
    iter_regex="step",
)

sim_constrained_profile = react_rs.profile(
    df=sim_result_constrained,
    partition_by="sim_id",
    iter_regex="step",
)
```

## **Development**

### **Build**

The [build workflow](.github/workflows/build.yml) is configured to build for Windows, Linux & MacOS :

- Build dependencies can be found in [Rust](Cargo.toml) & [Python](pyproject.toml) build specs.

- Outputs are stored as action pipeline artifacts

- An automated release is created for each new version of the app

### **Test**

Python test scripts are stored in the [tests](tests) directory, along with a sample dataset of 100k assets. 

These tests call all of the functionality within the native Python code & the Rust backend. At this stage there are no direct Rust tests, primarily due to issues with running cargo tests on Maturin / PyO3 projects. 

### **Notes**

- When working on [__init__.py](python/react_rs/__init__.py), your IDE may highlight imports from the react_rs package with an error - this is normal and can be ignored.
