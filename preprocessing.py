import pandas as pd
import numpy as np
import warnings


def convert_dummy_df(df,
                     drop_first=True,
                     drop_missing_rate=.9,
                     drop_unique_rate=.1,
                     na_replace=-999):
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


def convert_to_float(var, replace_missing=np.nan):
    """
        To do:
            This function replace '', '(null)', " " into numpy nan and transform var
            into float64 type
        param:
            var (numpy array): an numpy array of var
        """
    if var.dtype == np.float64 or var.dtype == np.float32 or var.dtype == np.float64:
        var = var.astype(np.float64)
        return var
    if var.dtype == np.int64 or var.dtype == np.int32 or var.dtype == np.int64:
        var = var.astype(np.int64)
        return var
    if var.dtype == np.datetime64:
        var = np.array([x - np.datetime64('0000-01-01T00:00') if str(x) != 'NaT' else np.nan for x in var])
        return var
    var_float = var.copy()
    var_float = [str(x) for x in var_float.flat]
    var_float[var_float == ''] = np.nan
    var_float[var_float == ' '] = np.nan
    var_float[var_float == '(null)'] = np.nan
    var_float = [s.replace('$', '').replace(',', '').replace(' ', '').replace('#', '') if str(s) != 'nan'
                 else np.nan for s in var_float]
    try:
        var_float = np.array([float(x) for x in var_float],
                             dtype='float64').reshape(var_float.shape)
        return var_float
    except:
        warnings.warn("Cannot transform to numpy float64.")
        var = np.array([str(x) if str(x) != 'nan' else replace_missing for x in var.flat], dtype='object')
        return var
