import numpy as np
import pandas as pd

from pandas import DataFrame as PandasDF

pd.options.mode.chained_assignment = None


def pivot_num(data: PandasDF, var: str, performance: str = 'bad_ind',
              n: int = 10, ks: float = True, max_ks_only: float = False) -> PandasDF:
    """
    Output the bad rate segment for a particular variable in group

    Parameter
        data: A pandas Dataframe, data
        var: A string, variable interested
        performance: A string, your performance variable.
        n: An integer, number of segmentation groups.
    Retrun:
        A dataframe contains varaible's name; level, bad rate, and count for each group.
    """
    temp = data.loc[:, [var, performance]].copy()
    temp_missing = temp.loc[temp[var].isnull(), :]
    temp_noMissing = temp.loc[~temp[var].isnull(), :]
    temp_noMissing.sort_values(var, inplace=True)
    length = round(temp_noMissing.shape[0]/n)

    group = temp_noMissing.groupby(np.arange(temp_noMissing.shape[0]) // length).apply(
        lambda obj: pd.Series({
            'var': var,
            'level': str(obj[var].min()) + ' - ' + str(obj[var].max()),
            'bad rate': obj[performance].mean(),
            'count': len(obj[performance])
        }))
    group_missing = pd.DataFrame({
            'var': var,
            'level': np.nan,
            'bad rate': temp_missing[performance].mean(),
            'count': temp_missing.shape[0],
            'ks': np.nan
        }, index=[n+1, ])
    # temp = group[['bad rate', 'count']].copy()
    if ks or max_ks_only:
        group['bad'] = [r * c for r, c in zip(group['bad rate'], group['count'])]
        group['cum_bad'] = [sum(group.loc[0:i, 'bad']) for i in range(group.shape[0])]
        group['cum_count'] = [sum(group.loc[0:i, 'count']) for i in range(group.shape[0])]
        group['cum_good'] = [c - b for c, b in zip(group['cum_count'], group['cum_bad'])]
        group['ks'] = [
            (100 * abs(g/group.loc[group.shape[0]-1, 'cum_good'] - b/group.loc[group.shape[0]-1, 'cum_bad']))
            for g, b in zip(group.cum_good, group.cum_bad)]
        max_index = group['ks'].idxmax()
        if max_ks_only:
            return group.loc[[max_index, ], ['var', 'ks']]
        group['ks'] = ['%.1f%%' % x for x in group['ks']]
        group = group.append(group_missing)
        group['bad rate'] = ['%.2f%%' % (x * 100) for x in group['bad rate']]

        group.style.applymap(highlight, subset=pd.IndexSlice[max_index, ['ks']])

        return group[['var', 'level', 'bad rate', 'count', 'cum_bad', 'cum_good', 'ks']]
    else:
        group = group.append(group_missing[['var', 'level', 'bad rate', 'count']])
        group.rename(columns={'bad rate': 'avg %s' % performance}, inplace=True)
        return group[['var', 'level', 'avg %s' % performance, 'count']]


def highlight(s: str) -> str:
    return 'background-color: yellow'
