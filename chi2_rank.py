import pandas as pd
import numpy as np
import warnings
from .attr_rank import attr_rank
from pandas import DataFrame as PandasDF
from numpy import array as Array
from scipy.stats import chi2_contingency
from typing import Any, List, Union


class chi2_rank(attr_rank):
    def __init__(self):
        attr_rank.__init__(self)

    @classmethod
    def chi2_contingency(cls, x: Union[Array, List[Any]],
                         y: Union[Array, List[Any]],
                         continuous_x: int = 10, continuous_y: int = 10):
        """
        To do:
            This function gives the chi-square independency test for two input variables
        Parameter:
            x (numpy array): first attribute
            y (numpy array): second attribute
            continuous_x (int): number of parts divided for x if x is continuous
            continuous_y (int): number of parts divided for y if y is continuous
        Return:
            chi2, p-value, degree-of-freedom, expected-frequency table
        """
        if type(y).__name__ == pd.Series.__name__:
            y = y.values
        if type(x).__name__ == pd.Series.__name__:
            x = x.values

        if len(x) != len(y):
            raise AttributeError("x and y don't have equal length.")
        elif (sum([str(e) != 'nan' for e in x.flat]) < round(len(x)/20) or
                sum([str(e) != 'nan' for e in y.flat]) < round(len(y)/20)):
            result = np.repeat(np.nan, 4)
        elif np.array_equal(x, y):
            result = np.repeat(-1, 4)
            warnings.warn('Array x and y are the same.')
        else:
            table = cls.contingency_table(x, y, continuous_x, continuous_y)
            if len(table) == 0:
                return np.repeat(np.nan, 4)
            result = chi2_contingency(table)
        result = dict(zip(['chi2', 'p_value', 'DOF', 'expected_freq'], result))
        return result

    def chi2_contingency_rank(self, X: PandasDF, y: Union[Array, List[Any]],
                              continuous_x: int = 10, continuous_y: int = 10,
                              num_of_top: int = 0):
        """
        To do:
            This function gives the chi-square independency test between y and each columns in X.
            The function also gives an option
            to list top related variables to each of the variables by doing a pairwise chi2_contigency test.
        Parameter:
            X (pandas DF): Independence variabels of interest
            y (np.array): dependence variable
            continuous_x (int): number of parts divided for x if x is continuous
            continuous_y (int): number of parts divided for y if y is continuous
            num_of_top: Option to show top related variables for each of the variables in X. Default is 0
        Return:
            output_df (pd.DataFrame): a pandas DF with X.columns in rows,
                    and chi2 p-value to performance variables, top related variabels in columns.
        """
        self.X = X
        self.y = y
        self.continuous_n = {'x': continuous_x, 'y': continuous_y}
        dim = X.shape[1]
        list_of_variables = X.columns.tolist()
        output_dict = {'var': list_of_variables,
                       'p_value': np.repeat(1.0, dim).tolist(),
                       'chi2': np.repeat(-1, dim).tolist(),
                       'DOF': np.repeat(-1, dim).tolist()}
        continuous_n = [continuous_x, continuous_y]
        for i in range(num_of_top):
            output_dict['top_related_variable%d' % (i+1)] = np.repeat(0, dim).tolist()
        output_df = pd.DataFrame(output_dict, index=range(dim), dtype=np.float64)
        for i in output_df.index:
            var_x = output_df.loc[i, 'var']
            chi_obj = self.chi2_contingency(X[var_x].values,
                                            y, *continuous_n)
            output_df.loc[i, ['p_value', 'chi2', 'DOF']] = pd.Series({
                'p_value': chi_obj['p_value'],
                'chi2': chi_obj['chi2'],
                'DOF': chi_obj['DOF']})
        if num_of_top > 0:
            independency_df = np.zeros([dim, dim])
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        independency_df[i, j] = np.nan
                    else:
                        var_x = output_df.loc[i, 'var']
                        var_y = output_df.loc[j, 'var']
                        independency_df[i, j] = self.chi2_contingency(
                            X[var_x].values,
                            X[var_y].values, *continuous_n)['p_value']
            independency_order = np.argsort(independency_df, axis=1)[np.arange(dim), 0:num_of_top]
            output_df[sorted([x for x in output_df.columns if x.find('top_') == 0])] = np.array(
                list_of_variables, dtype='object')[independency_order]
            # add p-value after num_of_top varaibles names
            ind_df = independency_df.copy()
            ind_df.sort(axis=1)
            ind_df = np.array(
                ["_p-values:%g" % p for p in np.array(ind_df.reshape(ind_df.size))],
                dtype='object').reshape((dim, -1))[np.arange(dim), 0:num_of_top]
            output_df[sorted([x for x in output_df.columns if x.find('top_') == 0])] += ind_df

        try:
            temp_rank = np.argsort(output_df['p_value'].tolist())
            output_df.loc[temp_rank, 'importance_rank'] = np.arange(dim)
            output_df = output_df.sort_values('importance_rank').reset_index(drop=True)
            return output_df[['importance_rank', 'var', 'p_value', 'chi2', 'DOF'] +
                             [x for x in output_df.columns if x.find('top_') == 0]]
        except ValueError:
            return output_df[['p_value', 'chi2', 'DOF']]
