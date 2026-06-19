# Test Rust app via Python library
import polars as pl
import pytest
import react_rs

# Configure test input
model_config = {
    "general": {"states": [15, 25], "values": [0.95, 0.4]},
    "short": {"base_model": "general", "mean_age": 15},
    "medium": {"base_model": "general", "mean_age": 25},
    "long": {"base_model": "general", "mean_age": 35},
}

wb = react_rs.Weibull()

for model, params in model_config.items():
    wb.fit(model_name=model, **params)

wb.generate()

test_case = [
    {
        "input_data": pl.read_parquet("./tests/data/input.parquet"),
        "id_col": "uuid",
        "age_col": "step_0",
        "cost_col": "value",
        "survival_curves": pl.DataFrame(
            {"model": wb.curves.keys(), "curve": wb.curves.values()}
        ),
        "curve_col": "curve",
        "n_sims": 50,
        "n_steps": 20,
        "parallel_limit": 10,
    }
]


# Test simulation in Rust via Python
@pytest.mark.parametrize(argnames="test_case", argvalues=test_case)
class TestReactRS:
    def test_sim(self, test_case):
        df_test: pl.DataFrame = test_case["input_data"].join(
            test_case["survival_curves"],
            "model",
            "left",
        )
        sim_result = react_rs.simulate(
            df=df_test,
            id_col=test_case["id_col"],
            age_col=test_case["age_col"],
            cost_col=test_case["cost_col"],
            probs_col=test_case["curve_col"],
            n_sims=test_case["n_sims"],
            n_steps=test_case["n_steps"],
            parallel_limit=test_case["parallel_limit"],
        )

        if not df_test.height * test_case["n_sims"] == sim_result.height:
            raise AssertionError("sim results do not contain expected number of rows")

        if (
            not len([i for i in sim_result.columns if i.startswith("step")])
            == test_case["n_steps"] + 1
        ):
            raise AssertionError(
                "sim results do not contain the expected number of time steps"
            )

    def test_constraint(self, test_case):
        df_test: pl.DataFrame = test_case["input_data"].join(
            test_case["survival_curves"],
            "model",
            "left",
        )
        sim_result = react_rs.simulate(
            df=df_test,
            id_col=test_case["id_col"],
            age_col=test_case["age_col"],
            cost_col=test_case["cost_col"],
            probs_col=test_case["curve_col"],
            n_sims=test_case["n_sims"],
            n_steps=test_case["n_steps"],
            parallel_limit=test_case["parallel_limit"],
        )

        # Constrain simulation output
        sim_result_constrained = react_rs.constrain(
            df=sim_result,
            constrain_steps=test_case["n_steps"],
            iter_regex="step",
            cost_col="cost",  # currently name is set to 'cost' by rust code
            constraints=[int(50e6) for _ in range(test_case["n_steps"])],
            partition_by="sim_id",
            parallel_limit=test_case["parallel_limit"],
        )

        if not df_test.height * test_case["n_sims"] == sim_result_constrained.height:
            raise AssertionError(
                "constrained results do not contain expected number of rows"
            )

        if (
            not len([i for i in sim_result_constrained.columns if i.startswith("step")])
            == test_case["n_steps"] + 1
        ):
            raise AssertionError(
                "constrained results do not contain the expected number of time steps"
            )

    def test_aggregation(self, test_case):
        df_test: pl.DataFrame = test_case["input_data"].join(
            test_case["survival_curves"],
            "model",
            "left",
        )
        sim_result = react_rs.simulate(
            df=df_test,
            id_col=test_case["id_col"],
            age_col=test_case["age_col"],
            cost_col=test_case["cost_col"],
            probs_col=test_case["curve_col"],
            n_sims=test_case["n_sims"],
            n_steps=test_case["n_steps"],
            parallel_limit=test_case["parallel_limit"],
        )

        # Aggregate simulation outputs
        sim_result_agg = react_rs.aggregate(
            df=sim_result,
            partition_by="sim_id",
            iter_regex="step",
            target_value=0,
            cost_col=None,
        )

        if not test_case["n_sims"] == sim_result_agg.height:
            raise AssertionError(
                "aggregation results do not contain expected number of rows"
            )

        # Aggregations ignore initnal age col (step_0) so steps should match 'n_steps'
        if (
            not len([i for i in sim_result_agg.columns if i.startswith("step")])
            == test_case["n_steps"]
        ):
            raise AssertionError(
                "aggregation results do not contain the expected number of time steps"
            )

    def test_profiler(self, test_case):
        df_test: pl.DataFrame = test_case["input_data"].join(
            test_case["survival_curves"],
            "model",
            "left",
        )
        sim_result = react_rs.simulate(
            df=df_test,
            id_col=test_case["id_col"],
            age_col=test_case["age_col"],
            cost_col=test_case["cost_col"],
            probs_col=test_case["curve_col"],
            n_sims=test_case["n_sims"],
            n_steps=test_case["n_steps"],
            parallel_limit=test_case["parallel_limit"],
        )

        # Age Profile across iterations & timesteps
        sim_profile = react_rs.profile(
            df=sim_result,
            partition_by="sim_id",
            iter_regex="step",
            parallel_limit=test_case["parallel_limit"],
        )

        if "value" not in sim_profile.columns:
            raise AssertionError("value (age) column not present in profile ouput")

        if (  # id's are 0-based so total == 'n_sims' - 1
            not sim_profile.select("sim_id").max().to_series().to_list()[0]
            == test_case["n_sims"] - 1
        ):
            raise AssertionError(
                "profile simulation count does not match input simulation count"
            )

        if (
            not len([i for i in sim_profile.columns if i.startswith("step")])
            == test_case["n_steps"] + 1
        ):
            raise AssertionError(
                "profile results do not contain the expected number of time steps"
            )
