import numpy as np
import pandas as pd
import math
from pandas import DataFrame as PandasDF
from numpy import array as Array
from typing import Dict, List, Union


def iv_rank(df: PandasDF, y: Union[List[float], Array],
            n: int = 10, min_split: int = 10, sorted: bool = True) -> PandasDF:
    """
    Rank the attributes using information value

    param:
        df: data with independent variables
        y: data with dependent variable
        n: number of split when doing woe
        min_split: min number of unique values to consider a variable continuous
        sorted: sort the output by iv in descending order
    """
    out = pd.DataFrame({
        'var': list(df.columns),
        'iv': np.repeat(np.nan, len(df.columns))},
        index=range(len(df.columns)))
    for idx in out.index:
        var = out.loc[idx, 'var']
        var = pd.to_numeric(df[var], errors='coerce')
        if var.nunique() >= min_split:
            out.loc[idx, 'iv'] = woe_iv(var, y, n=n)['iv']
        else:
            pass
    out['predictiveness'] = ['Not useful' if x < 0.02 else
                             'Weak' if 0.02 <= x < .1 else
                             'Medium' if 0.1 <= x < .3 else
                             'Strong' if 0.3 <= x < .5 else
                             'Suspicous' if x > .5 else 'NaN' for x in out.iv]
    if sorted:
        out = out.sort_values('iv', ascending=False).reset_index(drop=True)
    return out


def woe_iv(x: Union[List[float], Array],
           y: Union[List[float], Array], n: int = 10, detail: bool = False
           ) -> Dict[str, float]:
    """
    Get the woe and information value for two variables

    param:
        x: np.array, a continuous variable for woe calculating
        y: np.array, bad ind
        n: number of split, default is 10
    """
    df = pd.DataFrame({'x': np.array(x), 'y': np.array(y)}, index=range(len(x)))
    # points = [np.nanpercentile(x, 100/n*i) for i in range(n+1)]
    df['bins'] = pd.qcut(df['x'], q=n, duplicates='drop')
    df['bins'] = [str(x) for x in df.bins]
    df['labels'] = pd.qcut(df['x'], q=n, duplicates='drop', labels=False)
    out = df.groupby('bins').apply(lambda obj: pd.Series({
                    'labels': obj.labels.unique()[0],
                    'counts': len(obj),
                    'min': np.nanmin(obj.x),
                    'max': np.nanmax(obj.x),
                    'bad rate': "%.1f%%" % (np.nanmean(obj.y) * 100),
                    'bads': np.nansum(obj['y']),
                    'goods': len(obj) - np.nansum(obj['y']),
                    'percentage_bads': np.nansum(obj['y']) / np.nansum(df.y),
                    'percentage_goods': (len(obj) - np.nansum(obj['y'])) / (len(df) - np.nansum(df.y))
        })).reset_index()
    out['woe'] = [math.log(bads / goods) if goods != 0 and bads != 0 else 1 for bads, goods in zip(
        out.percentage_bads,
        out.percentage_goods)]
    out['iv'] = [(bads - goods) * woe for bads, goods, woe in zip(
        out.percentage_bads,
        out.percentage_goods, out.woe)]
    out = out.sort_values('labels').reset_index(drop=True)
    if not detail:
        out = out[['bins', 'labels', 'counts', 'min', 'max', 'bad rate', 'woe', 'iv']]
    iv = np.nansum(out.iv)
    return {'woe': out, 'iv': iv}
