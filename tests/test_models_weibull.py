# # Weibull Test Script
# import sys

# import numpy as np
# import polars as pl
# import pytest

# sys.path.append("..")

# from python.react_rs._weibull import Weibull as wb

# # Test cases
# test_case = [
#     {
#         "survival_test": {"x": 1, "shape": 1, "scale": 1, "result": 0.3679},
#         "apply_test": {
#             "shape": 1,
#             "scale": 1,
#             "max_value": 3,
#             "result": [0.3679, 0.3679, 0.3679],
#         },
#         "mean_test": {"shape": 0.5, "scale": 0.05, "result": 0.1},
#         "weibull_fit_test": {
#             "ages": [
#                 20,
#                 30,
#                 40,
#                 50,
#                 60,
#                 70,
#                 80,
#             ],
#             "values": [0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1],
#             "result": (2.1834, 61.6781),
#         },
#         "adjust_scale_test": {"shape": 0.5, "mean_age": 35, "result": (0.5, 17.5)},
#         "adjust_model_test": {
#             "self": wb(data=None),
#             "params": {"base": (0.5, 17.5)},
#             "base_model": "base",
#             "adjust_value": 45,
#             "result": (0.5, 22.5),
#         },
#         # "config_fit_run_test": {
#         #     "self": wb(data=None),
#         #     "data": "python/tests/data/test_inventory.csv",
#         #     "n_iters": 1,
#         #     "n_steps": 3,
#         #     "model_params": {
#         #         "general": {
#         #             "states": [15, 25],
#         #             "values": [0.95, 0.4],
#         #             "model_name": "general",
#         #         },
#         #         "short": {
#         #             "base_model": "general",
#         #             "adjust_value": 15,
#         #             "model_name": "short",
#         #         },
#         #         "medium": {
#         #             "base_model": "general",
#         #             "adjust_value": 25,
#         #             "model_name": "medium",
#         #         },
#         #         "long": {
#         #             "base_model": "general",
#         #             "adjust_value": 35,
#         #             "model_name": "long",
#         #         },
#         #     },
#         # },
#     }
# ]


# # Tests
# @pytest.mark.parametrize(argnames="test_case", argvalues=test_case)
# class TestWeibull:
#     def test_survival_function(self, test_case):
#         """Test for an expected result"""
#         test_output = wb.survival_function(
#             x=test_case["survival_test"]["x"],
#             shape=test_case["survival_test"]["shape"],
#             scale=test_case["survival_test"]["scale"],
#         )

#         if round(test_output, 4) != test_case["survival_test"]["result"]:
#             raise AssertionError(
#                 "test_survival_function failed : output does not sufficiently match test value"
#             )

#     def test_apply_survival_function(self, test_case):
#         """Test for an expected result"""
#         test = wb.apply_survival_function(
#             scale=test_case["apply_test"]["scale"],
#             shape=test_case["apply_test"]["shape"],
#             max_value=test_case["apply_test"]["max_value"],
#         )

#         if not np.all(
#             [
#                 round(i, 4) == test_case["apply_test"]["result"][n]
#                 for n, i in enumerate(test)
#             ]
#         ):
#             raise AssertionError(
#                 "test_apply_survival_funtion failed : output does not sufficiently match test value"
#             )

#     def test_weibull_mean(self, test_case):
#         """Test for an expected result"""
#         test = wb.weibull_mean(
#             shape=test_case["mean_test"]["shape"], scale=test_case["mean_test"]["scale"]
#         )

#         if test != test_case["mean_test"]["result"]:
#             raise AssertionError(
#                 "test_weibull_mean failed : output does not sufficiently match test value"
#             )

#     def test_fit_weibull_params(self, test_case):
#         """Test for an expected result"""
#         test = wb.fit_weibull_params(
#             ages=test_case["weibull_fit_test"]["ages"],
#             values=test_case["weibull_fit_test"]["values"],
#         )
#         test = tuple([round(i, 4) for i in test])

#         if test != test_case["weibull_fit_test"]["result"]:
#             raise AssertionError(
#                 "test_weibull_mean failed : output does not sufficiently match test value"
#             )

#     def test_adjust_scale_params(self, test_case):
#         """Test for an expected result"""
#         test = wb.adjust_scale_params(
#             shape=test_case["adjust_scale_test"]["shape"],
#             mean_age=test_case["adjust_scale_test"]["mean_age"],
#         )

#         if test != test_case["adjust_scale_test"]["result"]:
#             raise AssertionError(
#                 "test_adjust_scale_params failed : output does not sufficiently match test value"
#             )

#     def test_adjust_model(self, test_case):
#         """Test for an expected result"""
#         test_case["adjust_model_test"]["self"].params = test_case["adjust_model_test"][
#             "params"
#         ]

#         test = wb._adjust_model(
#             self=test_case["adjust_model_test"]["self"],
#             base_model=test_case["adjust_model_test"]["base_model"],
#             adjust_value=test_case["adjust_model_test"]["adjust_value"],
#             model_name="test",
#         )

#         if test.params["test"] != test_case["adjust_model_test"]["result"]:
#             raise AssertionError(
#                 "test_adjust_model failed : output does not sufficiently match test value"
#             )

#     # def test_config_fit_run_model(self, test_case):
#     #     """Test for an expected result"""
#     #     tst = test_case["config_fit_run_test"]["self"]
#     #     tst.data = pl.read_csv(test_case["config_fit_run_test"]["data"]).with_columns(
#     #         pl.lit(1).alias("model_iteration")
#     #     )
#     #     tst.data = tst.data.rename(
#     #         {
#     #             "category": "model",
#     #             "age": "state",
#     #             "asset_id": "id",
#     #         }
#     #     )

#     #     tst.fit(**test_case["config_fit_run_test"]["model_params"]["general"])

#     #     tst._adjust_model(**test_case["config_fit_run_test"]["model_params"]["short"])
#     #     tst._adjust_model(**test_case["config_fit_run_test"]["model_params"]["medium"])
#     #     tst._adjust_model(**test_case["config_fit_run_test"]["model_params"]["long"])

#     #     tst.run(
#     #         n_iters=test_case["config_fit_run_test"]["n_iters"],
#     #         n_steps=test_case["config_fit_run_test"]["n_steps"],
#     #     )

#     #     if (
#     #         f"""step_{test_case["config_fit_run_test"]["n_steps"]}"""
#     #         in tst.data.columns
#     #     ):
#     #         pass
#     #     else:
#     #         raise AssertionError(
#     #             "test_config_fit_run_model failed : expected columns not created"
#     #         )
