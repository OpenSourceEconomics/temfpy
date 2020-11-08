"""Tests for econometrics module."""
import pytest
import numpy as np
import pandas as pd
import temfpy.econometrics as tpe


@pytest.fixture
def data_generation(n_obs, n_var, choices, seed, beta_low=-3, beta_high=3):
    r"""Multinomial probit model. Simple Random DGP.

    .. math::
    Y_i^1* &= X_i^T \beta_1  + \varepsilon_1 \\ 
    Y_i^2* &= X_i^T \beta_2 + \varepsilon_2 \\
    \hdots \\
    Y_i^m* &= X_i^T \beta_m + \varepsilon_m \\
    Y_i &= \max \{Y_i^1*, Y_i^2*, \dots, Y_i^m*\} \\
    \beta_j \sim unif(0,1) \\
    X_i \sim \mathcal{N}(\bmp 0, \bmp I_{n\_var})

    Parameters
    ----------
    n_obs : int
            number of observations
    n_var : int
            number of exogenous variables
    choices : int 
              number of possible discrete outcomes 
    beta_low : float, optional 
               The default is -3
    beta_high : float, optional
                The default is 3

    Returns
    -------
    df: pandas.DataFrame
    Data frame containing the endogenous variable Y and the exogenous variables.
    """
    np.random.seed(seed)
    X = np.random.normal(size=n_obs * n_var).reshape((n_obs, n_var))
    betas = np.random.uniform(
        low=beta_low, high=beta_high, size=choices * n_var
    ).reshape(n_var, choices)
    Y_latent = np.dot(X, betas) + np.random.normal(size=n_obs * choices).reshape(
        n_obs, choices
    )
    Y = np.argmax(Y_latent, axis=1) + 1
    columns = [f"v{num}" for num in range(n_var)]
    df = pd.DataFrame(X, columns=columns)
    df["Y"] = Y

    return df


def test_multinomial_probit():
    n_obs_strategy = np.random.randint(50, 501)
    n_var_strategy = np.random.randint(2, 6)
    choices = np.random.randint(2, 6)
    cov_strategy = ["iid", "free"][np.random.randint(0, 2)]
    integration_strategy = [
        "mc_integration",
        "smc_integration",
        "gauss_integration",
    ][np.random.randint(0, 3)]
    algorithm_strategy = [
        "scipy_lbfgsb",
        "scipy_slsqp",
    ][np.random.randint(0, 2)]
    data, betas = data_generation(n_obs_strategy, n_var_strategy, choices, seed=10)
    all_columns = "+".join(data.columns.difference(["Y"]))
    formula = "Y~" + all_columns
    tpe.multinomial_probit(
        formula, data, cov_strategy, integration_strategy, algorithm_strategy
    )
