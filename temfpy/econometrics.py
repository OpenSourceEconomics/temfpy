"""Econometric Methods.
We provide a variety of econometric methods used in data science.
"""
import numpy as np
import pandas as pd
import patsy
import temfpy.integration_methods
from estimagic.optimization.optimize import maximize


def multinomial_processing(formula, data, cov_structure):
    r"""Construct the inputs for the multinomial probit function.

    .. math::

    Parameters
    ----------
        formula : str
                  A patsy formula comprising the independent variable and the dependent variables.

    data : pd.DataFrame
           A pandas data frame with shape :math:`n_obs \times n_var + 1`.

        cov_structure : str
                        Available options are 'iid' or 'free'.

    Returns:
    --------
    y : np.array
        1d numpy array of shape n_obs with the observed choices.

    x : np.array
        2d numpy array of shape :math:'(n_obs, n_var)' including the independent variables.
    params_df : pd.Series
                The data are naive starting values for the parameters.

    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    y, x = patsy.dmatrices(formula, data, return_type="dataframe")
    data = pd.concat([y, x], axis=1).dropna()
    y, x = patsy.dmatrices(formula, data, return_type="dataframe")

    n_var = len(x.columns)
    n_choices = len(np.unique(y.to_numpy()))

    np.random.seed(1998)
    bethas = np.random.rand(n_var * (n_choices - 1)) * 0.1

    if cov_structure == "iid":

        index_tuples = []
        var_names = list(x.columns)
        for choice in range(n_choices - 1):
            index_tuples += [
                ("choice_{}".format(choice), "betha_{}".format(name))
                for name in var_names
            ]

        start_params = bethas

    else:
        covariance = np.eye(n_choices - 1)
        cov = []
        for i in range(n_choices - 1):
            for j in range(n_choices - 1):
                if j <= i:
                    cov.append(covariance[i, j])

        cov = np.asarray(cov)

        index_tuples = []
        var_names = list(x.columns)
        for choice in range(n_choices - 1):
            index_tuples += [
                ("choice_{}".format(choice), "betha_{}".format(name))
                for name in var_names
            ]

        j = (n_choices) * (n_choices - 1) / 2
        index_tuples += [("covariance", i) for i in range(int(j))]

        start_params = np.concatenate((bethas, cov))

    params_sr = pd.Series(
        data=start_params, index=pd.MultiIndex.from_tuples(index_tuples), name="value"
    )

    y = y - y.min()

    return (
        y.to_numpy(dtype=np.int64).reshape(len(y)),
        x.to_numpy(dtype=np.float64),
        params_sr,
    )


def multinomial_probit_loglikeobs(params, y, x, cov_structure, integration_method):
    r"""Individual log-likelihood of the multinomial probit model.

    .. math::

    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent variable and the independent variables.

    y : np.array
        1d numpy array of shape :math:'n_obs' with the observed choices

    x : np.array
        2d numpy array of shape :math:'(n_obs, n_var)' including the independent variables.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         'mc_integration', 'smooth_mc_integration', ...


    Returns:
    --------
        loglikeobs : np.array
                     1d numpy array of shape :math:'(n_obs)' with likelihood contribution.

    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    n_var = np.shape(x)[1]
    n_choices = len(np.unique(y))

    if cov_structure == "iid":
        cov = np.eye(n_choices - 1) + np.ones((n_choices - 1, n_choices - 1))

    else:
        covariance = params["covariance"].to_numpy()

        cov = np.zeros((n_choices - 1, n_choices - 1))

        a = 0
        for i in range(n_choices - 1):
            k = i + 1
            cov[i, : (i + 1)] = covariance[a : (a + k)]
            a += k

        for i in range(n_choices - 1):
            cov[i, (i + 1) :] = cov[(i + 1) :, i]

    bethas = np.zeros((n_var, n_choices))

    for i in range(n_choices - 1):
        bethas[:, i] = params["choice_{}".format(i)].to_numpy()

    u_prime = x.dot(bethas)

    choice_prob_obs = getattr(temfpy.integration_methods, integration_method)(
        u_prime, cov, y
    )

    choice_prob_obs[choice_prob_obs <= 1e-250] = 1e-250

    loglikeobs = np.log(choice_prob_obs)

    return loglikeobs


def multinomial_probit_loglike(params, y, x, cov_structure, integration_method):
    r"""log-likelihood of the multinomial probit model.

    .. math::


    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent variable and the independent variables.

    y : np.array
        1d numpy array of shape :math:'n_obs' with the observed choices

    x : np.array
        2d numpy array of shape :math:'(n_obs, nvar)' including the independent variables.

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         'mc_integration', 'smooth_mc_integration', ...


    Returns:
    --------
        loglike : float
                  The value of the log-likelihood function evaluated at the given parameters.

    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    return multinomial_probit_loglikeobs(
        params, y, x, cov_structure, integration_method
    ).sum()


def multinomial_probit(formula, data, cov_structure, integration_method, algorithm):
    r"""Multinomial probit model.

    .. math::
        u_{ij} = x'_{ij} \beta_j + \varepsilon_{ij},..


    Parameters
    ----------
    formula : str
              A patsy formula comprising the dependent and the independent variables.

    data : pd.DataFrame
           A pandas data frame with shape

    cov_structure : str
                    Available options are 'iid' or 'free'.

    integration_method : str
                         'mc_integration', 'smooth_mc_integration',...

    algorithm : str
                Available options are 'scipy_L-BFGS-B',...


    Returns
    -------
    result_dict: dic
    Information of the optimization.

    params: Parameters :math:'\beta_j' that minimize the log-likelihood function.

    Notes
    -----

    References
    ----------
    ..[G1994] Geweke, J., Keane, M., and Runkle, D. 1994.
      Alternative Computational Approaches to Inference in the Multinomial Probit Model.
      The MIT Press.

    Examples
    --------
    """

    y, x, params = multinomial_processing(formula, data, cov_structure)
    params_df = pd.DataFrame(params, columns=["value"])

    if cov_structure == "iid":
        constraints = []

    else:
        constraints = [
            {"loc": "covariance", "type": "covariance"},
            {"loc": ("covariance", 0), "type": "fixed", "value": 1.0},
        ]

    result = maximize(
        multinomial_probit_loglike,
        params_df,
        algorithm,
        criterion_kwargs={
            "y": y,
            "x": x,
            "cov_structure": cov_structure,
            "integration_method": integration_method,
        },
        constraints=constraints,
        dashboard=True,
    )

    return result
