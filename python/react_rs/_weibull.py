# Weibull Survival Curve Generation Module
from math import gamma
from typing import Dict, List, Self, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


class Weibull:
    """Fit Weibull models & generate survival curves..."""

    def __init__(self):
        self.models: Dict[str, Tuple[float, float]] = {}
        self.curves: Dict[str, List[float]] = {}

    @staticmethod
    def apply_survival_function(shape, scale, max_value=200) -> List[float]:
        """
        Applies the Weibull survival function to dataset.  Default output provides the conditional
        survival probability - given survived to time x, probability of surviving to time x+1. If
        shape and scale argument are not provided, defaults to class shape/scale attributes.
        """

        # calculate survival probability over value range
        probs = np.exp(-((np.arange(max_value + 1) / scale) ** shape))

        # convert probability to conditional, given survival to previous value
        return list(np.concatenate(([probs[1]], probs[2:] / (probs[1:-1] + 0.0000001))))

    @staticmethod
    def fit_weibull_model(ages, values) -> Tuple[float, float]:
        """
        Performs linear regression on survival ages, values to estimate shape and scale parameters.
        Output is returned a tuple.
        """
        # transform the survival parameters and fit a linear model
        transformed_ages = np.log(np.array(ages)).reshape((-1, 1))
        transformed_values = np.log(np.log(1 / np.array(values)))
        model = LinearRegression().fit(transformed_ages, transformed_values)

        # extract shape and scale parameters
        shape = float(model.coef_[0])
        scale = np.exp(round(model.intercept_) / -shape)

        return (shape, scale)

    def fit(
        self,
        model_name: str | None = None,
        states: List[int] | None = None,
        values: List[float] | None = None,
        base_model: str | None = None,
        mean_age: int | None = None,
    ) -> Self:
        """
        Generates fit parameters for supplied model configuration, and allows users
        to create additional model variations based on an adjusted mean.

        To generate a base model, users should supply :
            - model_name
            - states
            - values

        To adjust a configured base model to a new mean survival age, users should supply :
            - base_model
            - mean_age

        Inputs
        ---
        - model_name: string name for base model configuration
        - states : list[int] representing survival ages
        - values : list[int] representing survival probabilities
        - base_model : string name of model configuration to apply adjusted mean survival age
        - mean_age : int value of mean survival age to be used to adjust a base model

        Returns
        ---
        self

        """
        # Configure a base model
        if (
            isinstance(model_name, str)
            and isinstance(states, list)
            and isinstance(values, list)
        ):
            self.models[model_name] = self.fit_weibull_model(ages=states, values=values)

        # Configure an adjusted model
        elif base_model not in self.models.keys():
            raise ValueError(
                f"Selected base_model has not been configured : {base_model}"
            )

        elif (
            isinstance(base_model, str)
            and isinstance(model_name, str)
            and isinstance(mean_age, int)
        ):
            shape = self.models[base_model][0]
            self.models[model_name] = (shape, mean_age / gamma(1 + 1 / shape))

        else:
            raise ValueError("Incorrect parameters passed to fit method...")

        return self

    def generate(self, n_steps: int = 200) -> Self:
        """
        Generate survival curves for configured models

        Input
        ---
        - n_steps : int representing the number of desired steps in each survival curve

        Returns
        ---
        self
        """
        if len(self.models) == 0:
            raise AttributeError("""At least one model should be configured
                via Weibull.fit() before running Weibull.generate()...
            """)

        for model, params in self.models.items():
            self.curves[model] = [
                float(x)
                for x in self.apply_survival_function(
                    shape=params[0],
                    scale=params[1],
                    max_value=n_steps,
                )
            ]

        return self
