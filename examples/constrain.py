# Example use of Rust app via Python library
from time import time

import polars as pl
import react_rs

# Execute Rust from Python
output_location = "examples/data/output"

start = time()
const = react_rs.constrain(
    df=pl.read_parquet(f"{output_location}/base_sim.parquet"),
    constrain_steps=30,
    iter_regex="step",
    cost_col="cost",
    constraints=[int(50e6) for _ in range(30)],
    partition_by="sim_id",
    parallel_limit=10,
)

sim_end = time()
print(f"Simulation complete in {round(sim_end - start, 1)}s")

print(const.filter(pl.col("sim_id") == 1))
const.write_parquet(f"{output_location}/const_sim.parquet", compression_level=10)
del const

# Aggregation
start = time()
const_agg = react_rs.aggregate(
    df=pl.read_parquet(f"{output_location}/const_sim.parquet"),
    partition_by="sim_id",
    iter_regex="step",
    target_value=0,
    cost_col="cost",
)

sim_end = time()
print(f"Aggregation complete in {round(sim_end - start, 1)}s")

print(const_agg.filter(pl.col("sim_id") == 1))
const_agg.write_csv(f"{output_location}/const_agg.csv")
del const_agg
