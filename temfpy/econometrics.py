import numpy as np
import pandas as pd
import patsy
#import temfpy.integration_methods  
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
                The data are naive starting values for the parameters. The index contains the parameter names.
    
    Notes
    -----
    
    References
    ----------
    
    Examples
    --------      
    """
    y, x = patsy.dmatrices(formula, data, return_type='dataframe')
    data = pd.concat([y, x], axis=1).dropna()
    y, x = patsy.dmatrices(formula, data, return_type='dataframe')
    
    n_var = len(x.columns)
    n_choices = len(np.unique(y.to_numpy()))
    
    np.random.seed(1998)
    bethas = np.random.rand(n_var*(n_choices - 1))*0.1
    
    if cov_structure == 'iid':
    
        index_tuples = []
        var_names = list(x.columns) 
        for choice in range(n_choices - 1):
            index_tuples += [('choice_{}'.format(choice), 
                              'betha_{}'.format(name)) for name in var_names]
    
        start_params = bethas
        
    else:
        covariance = np.eye(n_choices - 1)
        cov = []
        for i in range(n_choices - 1):
            for j in range(n_choices - 1):
                if j<=i:
                    cov.append(covariance[i, j])
    
        cov = np.asarray(cov)
    
        index_tuples = []
        var_names = list(x.columns) 
        for choice in range(n_choices - 1):
            index_tuples += [('choice_{}'.format(choice), 
                              'betha_{}'.format(name)) for name in var_names]
    
        j = (n_choices) * (n_choices - 1) /2    
        index_tuples += [('covariance', i) for i in range(int(j))]
    
        start_params = np.concatenate((bethas, cov))
    
    params_sr = pd.Series(data=start_params, 
                          index=pd.MultiIndex.from_tuples(index_tuples), name='value')
    
    y = y - y.min()
        
    return y.to_numpy(dtype=np.int64).reshape(len(y)), x.to_numpy(dtype=np.float64), params_sr
