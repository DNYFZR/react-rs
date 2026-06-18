# Weibull Module Test Script
import numpy as np
import pytest
from react_rs import Weibull

test_case = [
    {
        "apply_test": {
            "shape": 1,
            "scale": 1,
            "max_value": 3,
            "result": [0.3679, 0.3679, 0.3679],
        },
        "weibull_fit_test": {
            "ages": [
                20,
                30,
                40,
                50,
                60,
                70,
                80,
            ],
            "values": [0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1],
            "result": (2.1834, 61.6781),
        },
        "fit_test": {
            "general": {"states": [15, 25], "values": [0.95, 0.4]},
            "short": {"base_model": "general", "mean_age": 15},
            "medium": {"base_model": "general", "mean_age": 25},
            "long": {"base_model": "general", "mean_age": 35},
        },
        "generate_test": [
            0.9999999987583741,
            0.9999998391816312,
            0.9999993503320288,
            0.9999974097946032,
            0.9999920742067548,
            0.9999802520307456,
            0.9999575000719333,
            0.9999178364449404,
            0.9998535661902077,
            0.9997551173957419,
            0.9996108866557849,
            0.9994070933786795,
            0.9991276429961413,
            0.9987539995992591,
            0.9982650689710078,
            0.997637093421585,
            0.9968435602725418,
            0.9958551262835377,
            0.9946395607724899,
            0.993161710643317,
        ],
    }
]


@pytest.mark.parametrize(argnames="test_case", argvalues=test_case)
class TestWeibull:
    def test_apply_survival_function(self, test_case):
        """Test for an expected result"""
        test = Weibull().apply_survival_function(
            scale=test_case["apply_test"]["scale"],
            shape=test_case["apply_test"]["shape"],
            max_value=test_case["apply_test"]["max_value"],
        )

        if not np.all(
            [
                round(i, 4) == test_case["apply_test"]["result"][n]
                for n, i in enumerate(test)
            ]
        ):
            raise AssertionError(
                "test_apply_survival_funtion failed : output does not sufficiently match test value"
            )

    def test_fit_weibull_params(self, test_case):
        """Test for an expected result"""
        test = Weibull().fit_weibull_model(
            ages=test_case["weibull_fit_test"]["ages"],
            values=test_case["weibull_fit_test"]["values"],
        )
        test = tuple([round(i, 4) for i in test])

        if test != test_case["weibull_fit_test"]["result"]:
            raise AssertionError(
                "test_weibull_mean failed : output does not sufficiently match test value"
            )

    def test_fit(self, test_case):
        test = {
            "general": (5.6433615369537735, np.float64(24.27843025275436)),
            "short": (5.6433615369537735, 16.224161817423703),
            "medium": (5.6433615369537735, 27.04026969570617),
            "long": (5.6433615369537735, 37.85637757398864),
        }
        res = Weibull()

        # Generate model fit for all configuratiosn
        for model, params in test_case["fit_test"].items():
            res.fit(model_name=model, **params)

        if res.models != test:
            raise AssertionError("Model fit test failed...")

    def test_generate(self, test_case):
        # Generate survival curves for all models
        res = Weibull()

        # Generate model fit for all configuratiosn
        for model, params in test_case["fit_test"].items():
            res.fit(model_name=model, **params)

        # Generate survival curves for all configurations
        res.generate()
        survival_curve = res.curves["long"][:20]

        if survival_curve != test_case["generate_test"]:
            raise AssertionError("Generate survival curves test failed...")
