# Weibull Simulation Module
import math
import random as rd

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression


class Weibull:
    """Cofigure, fit & execute a Weibull Simulation"""

    def __init__(self, n_iters: int | None = None):
        self.params = {}
        self.probabilities = pl.DataFrame
        self.n_iters = 1 if n_iters is None or n_iters < 1 else n_iters

    @staticmethod
    def survival_function(x, shape, scale):
        """
        Weibull survival function.  The survival function is the probability
        that the variate takes a value greater than x.
        """
        return np.exp(-((x / scale) ** shape))

    @staticmethod
    def apply_survival_function(shape, scale, max_value=200, cond_prob=True):
        """
        Applies the Weibull survival function to dataset.  Default output provides the conditional
        survival probability - given survived to time x, probability of surviving to time x+1. If
        shape and scale argument are not provided, defaults to class shape/scale attributes.
        """

        # calculate survival probability over value range
        probs = np.arange(max_value + 1)
        probs = Weibull.survival_function(probs, shape=shape, scale=scale)

        # convert probability to conditional, given survival to previous value
        if cond_prob:
            probs = np.concatenate(([probs[1]], probs[2:] / (probs[1:-1] + 0.0000001)))

        return probs

    @staticmethod
    def fit_weibull_params(ages, values):
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

    @staticmethod
    def adjust_scale_params(shape, mean_age):
        """
        Adjust shape parameter using mean age value.
        """
        return (shape, mean_age / math.gamma(1 + 1 / shape))

    def _adjust_model(self, base_model, adjust_value, model_name):
        """
        This is currently a method that is used to support the asset replacement
        model that adjusts the original weibull fit but adjusting the scale based
        on a revised mean age.  This has been generalised to support adjustments
        for other models, but will not have any impact on non-weibull methods.
        Parameters :
            - base_model (str) : name of the base model that will be used to create modified model
        adjust_value (int) : the value to be used to adjust the model
        model_name (str) : name of the new model
        Returns :
            - saved to model_params attribute - dictionary entry with key=model_name, value=model added
        """
        shape = self.params[base_model][0]
        self.params[model_name] = self.adjust_scale_params(
            shape=shape, mean_age=adjust_value
        )

        return self

    def survival_curve(
        self, states, values=None, model_name: str = "default", uncertainty=0.1
    ):
        """
        Fits/loads a model that will be used for simulation.  The input values will
        vary depending on the method specified as part of the constructor. This is
        currently configured for two methods:

        The initial method call will create and assign a dictionary to the model attribute.
        Subsequent method calls will add or modify existing elements. There is no method
        to remove a model at this time.

        Parameters :
            - states : array_like - data containing survival ages
            - values : array_like - data containing survival probability at given state
            - model name : string/int (optional) - name to be assigned to the model
        Returns :
            - saved to model_params attribute - dictionary entry with key=model_name, value=model added
        """

        """
        Configures model_params for use in the simulator.  This is completed individually for each
        model added to model_params.  The n_iters argument specifies the number of versions of each
        model to generate, each being modified by the uncertainty argument.

        Subsequent calls to this method will overwrite previous calls.

        Parameters :
            - n_iters (int) : number of model iterations, if set to
        adjust (float) : range to +/- for varying shape / scale parameters for each iteration
        Returns :
            - saved to probs attribute - pyspark.sql.dataframe.DataFrame
        """

        # Fit model params
        model = self.fit_weibull_params(ages=states, values=values)

        if self.params is None:
            self.params = {model_name: model}
        else:
            self.params[model_name] = model

        # Configure parameters
        probs = []
        for key, value in self.params.items():
            for j in range(1, self.n_iters + 1):
                if j == 1:
                    shape_iter = value[0]
                    scale_iter = value[1]
                    probs = [
                        {
                            "model_iteration": j,
                            "model": key,
                            "prob": [
                                float(x)
                                for x in self.apply_survival_function(
                                    shape=shape_iter, scale=scale_iter
                                )
                            ],
                        }
                    ]
                else:
                    shape_iter = (
                        value[0] + rd.uniform(-uncertainty, uncertainty) * value[0]
                    )
                    scale_iter = (
                        value[1] + rd.uniform(-uncertainty, uncertainty) * value[1]
                    )
                    probs.append(
                        {
                            "model_iteration": j,
                            "model": key,
                            "prob": [
                                float(x)
                                for x in self.apply_survival_function(
                                    shape=shape_iter, scale=scale_iter
                                )
                            ],
                        }
                    )

            # Configure DataFrame
            if list(self.params)[0] == key:
                self.probabilities = pl.DataFrame(probs)

            elif isinstance(self.probabilities, pl.DataFrame):
                self.probabilities = pl.concat(
                    [self.probabilities, pl.DataFrame(probs)]
                )

            else:
                raise TypeError("probs attribute should be of type List or DataFrame")

        return self


if __name__ == "__main__":
    model_config = {
        "general": {"states": [15, 25], "values": [0.95, 0.4], "model_name": "general"},
        "short": {"base_model": "general", "adjust_value": 15, "model_name": "short"},
        "medium": {"base_model": "general", "adjust_value": 25, "model_name": "medium"},
        "long": {"base_model": "general", "adjust_value": 35, "model_name": "long"},
    }

    check = Weibull().survival_curve(**model_config["general"])
    print(check.probabilities)

    for model, params in model_config.items():
        if model != "general":
            check._adjust_model(**params)
            print(check.params)
            print(check.probabilities)
