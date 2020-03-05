import pandas as pd
import numpy as np
import warnings
from numpy import array as Array
from pandas import DataFrame as PandasDF
from typing import Any, List, Union


def convert_dummy_df(df: PandasDF,
                     drop_first: PandasDF = True,
                     drop_missing_rate: float = .9,
                     drop_unique_rate: float = .1,
                     na_replace: int = -999) -> PandasDF:
    for i in df.columns.tolist():
        df[i] = convert_to_float(df[i].values)
        if df[i].dtype == 'O' and df[i].nunique() > drop_unique_rate * len(df[i]):
            df = df.drop(i, axis=1)
    df_dummy = pd.get_dummies(df, dummy_na=True, drop_first=drop_first)

    """Deal with missing values"""
    kk = df_dummy.isnull().sum()

    if any(kk > 0):
        # remove variables almost all missing
        near_all_missing = kk[kk > drop_missing_rate * len(df)].index
        df_dummy = df_dummy.loc[:, [x for x in df_dummy.columns
                                    if x not in near_all_missing]]
        kk = df_dummy.isnull().sum()
        if any(kk > 0):
            missing_col = kk[kk > 0].index
            for i in missing_col:
                # for those variables with missing values, add a new column for missings and
                # replace the missings
                df_dummy[i + '_missing'] = [1 if str(x) == 'nan' else 0
                                            for x in df[i]]
                df_dummy[i] = [na_replace if str(x) == 'nan' else x
                               for x in df_dummy[i]]

    if any(df_dummy.isnull().sum()) > 0:
        warnings.warn('Result still has missing values.')
    return df_dummy


def convert_to_float(var: Union[List[Any], Array], replace_missing: float = np.nan) -> Array:
    """
    This function replace '', '(null)', " " into numpy nan and transform var
    into float64 type

    param:
        var (numpy array): an numpy array of var
    """
    if var.dtype == np.datetime64:
        var = np.array([x - np.datetime64('0000-01-01T00:00') if str(x) != 'NaT' else np.nan for x in var])
        return var
    var_float = [str(x) for x in var]
    var_float = [s.replace('$', '').replace(',', '').replace(' ', '').replace('#', '') if str(s) != 'nan'
                 else np.nan for s in var_float]
    return pd.to_numeric(var_float, errors='coerce')
