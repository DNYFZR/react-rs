# Example use of Rust app via Python library
from time import time

import polars as pl
import react_rs

# Generate survival curve
model_config = {"states": [15, 25], "values": [0.95, 0.4], "model_name": "general"}

model_config_adjusted = {
    "short": {"base_model": "general", "adjust_value": 15, "model_name": "short"},
    "medium": {"base_model": "general", "adjust_value": 25, "model_name": "medium"},
    "long": {"base_model": "general", "adjust_value": 35, "model_name": "long"},
}

survival_curve = (
    react_rs.Weibull()
    .survival_curve(**model_config)
    .probabilities.select("prob")
    .to_series()
    .to_list()[0]
)

# Execute Rust from Python
output_location = "examples/data/output"

start = time()
res = react_rs.simulate(
    df=pl.read_parquet("examples/data/demo_input.parquet"),
    id_col="uuid",
    age_col="step_0",
    cost_col="value",
    probabilities=survival_curve,
    n_sims=500,
    n_steps=50,
    parallel_limit=10,
)

sim_end = time()
print(f"Simulation complete in {round(sim_end - start, 1)}s")

print(res.filter(pl.col("sim_id") == 1))
res.write_parquet(f"{output_location}/base_sim.parquet", compression_level=10)
del res

# Aggregation
start = time()
base_agg = react_rs.aggregate(
    df=pl.read_parquet(f"{output_location}/base_sim.parquet"),
    partition_by="sim_id",
    iter_regex="step",
    target_value=0,
    cost_col="cost",
)

sim_end = time()
print(f"Aggregation complete in {round(sim_end - start, 1)}s")

print(base_agg.filter(pl.col("sim_id") == 1))
base_agg.write_csv(f"{output_location}/base_agg.csv")
del base_agg
