"""Econometric Methods.
We provide a variety of econometric methods used in data science.
"""
import numpy as np


def mc_integration(u_prime, cov, y, n_draws=None):
    r"""Calculate probit choice probabilities with Monte-Carlo Integration.

    .. math::

    Parameters
    ----------
    u_prime : np.array
              2d array of shape  comprising the deterministic part of utilities

    cov : np.array
          2d array of shape

    y : np.array
        1d array of shape with the observed choices.

    n_draws : int
              Number of draws for Monte-Carlo integration.

    Returns:
    --------
    choice_prob_obs : np.array
                      1d array of shape  comprising the choice probabilities.

    Notes
    -----

    References
    ----------

    Examples
    --------

    """
    n_obs = np.shape(u_prime)[0]
    n_choices = np.shape(u_prime)[1]

    if n_draws is None:
        n_draws = n_choices * 2000

    np.random.seed(1995)
    base_error = np.random.normal(size=(n_obs * n_draws, (n_choices - 1)))
    chol = np.linalg.cholesky(cov)
    errors = chol.dot(base_error.T)
    errors = errors.T.reshape(n_obs, n_draws, (n_choices - 1))
    extra_column_errors = np.zeros((n_obs, n_draws, 1))
    errors = np.append(errors, extra_column_errors, axis=2)

    u = u_prime.reshape(n_obs, 1, n_choices) + errors

    index_choices = np.argmax(u, axis=2)

    choices = np.zeros((n_obs, n_draws, n_choices))
    for i in range(n_obs):
        for j in range(len(index_choices[1])):
            choices[i, j, int(index_choices[i, j])] = 1

    choice_probs = np.average(choices, axis=1)

    choice_prob_obs = choice_probs[range(len(y)), y]

    return choice_prob_obs
