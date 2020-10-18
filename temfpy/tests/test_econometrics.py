"""Tests for econometrics module."""

import numpy as np
import pandas as pd
import patsy
from estimagic.optimization.optimize import maximize
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers

import temfpy.integration_methods
from temfpy.econometrics import multinomial_processing
from temfpy.econometrics import multinomial_probit_loglikeobs
from temfpy.econometrics import multinomial_probit_loglike
from temfpy.econometrics import multinomial_probit


def data_generation(n_obs, n_var, choices, beta_low = -3, beta_high = 3):
    r"""Multinomial probit model. Simple Random DGP.

    .. math::
    Y_i^1* &= X_i^T \beta_1  + \varepsilon_1 \\ 
    Y_i^2* &= X_i^T \beta_2 + \varepsilon_2 \\
    \hdots \\
    Y_i^m* &= X_i^T \beta_m + \varepsilon_m \\
    Y_i &= \max \{Y_i^1*, Y_i^2*, \dots, Y_i^m*\}
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
    X = np.random.normal(size = n_obs*n_var).reshape((n_obs, n_var))
    betas = np.random.uniform(low = beta_low, high = beta_high, size = choices*n_var).reshape(n_var, choices)
    Y_latent = np.dot(X, betas) + np.random.normal(size = n_obs*choices).reshape(n_obs, choices)
    Y = np.argmax(Y_latent, axis = 1) + 1
    columns = [f'v{num}' for num in range(n_var)]
    df = pd.DataFrame(X, columns = columns)
    df['Y'] = Y
    
    return df

cov_strategy = arrays(np.str, 1, elements=["iid", "free"])
integration_strategy = arrays(np.str, 1, elements=["mc_integration", "smooth_mc_integration", "gauss_integration"])
algorithm_strategy = arrays(np.str, 1, elements=["scipy_L-BFGS-B", "scipy_SLSQP", "nlopt_bobyqa", "nlopt_newuoa_bound"])
n_obs_strategy = arrays(np.int, 1, elements=integers(50,500))
n_var_strategy = arrays(np.int, 1, elements=integers(2,10))
choices = arrays(np.int, 1, elements=integers(2,7))

strategy = (n_obs_strategy, n_var_strategy , choices, cov_strategy, integration_strategy, algorithm_strategy)

@given(*strategy)
def test_multinomial_probit(n_obs_strategy, n_var_strategy , choices, cov_structure, integration_method, algorithm): 
    data = data_generation(n_obs_strategy, n_var_strategy , choices)
    all_columns = "+".join(data.columns.difference(["Y"]))
    formula = "Y~" + all_columns
    multinomial_probit(formula, data, cov_structure, integration_method, algorithm)
    
    