import matplotlib.ticker as ticker
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame as PandasDF
from model_f import pivot_num
from typing import List, Union


def model_ks_plot(data: PandasDF, var_list: List[str], label: int = ['Risk Model', ],
                  performance: int = 'bad_ind',
                  title: str = 'Model Performance',
                  line_size: int = 1,
                  title_font_size: int = 18,
                  x_label_size: int = 12, y_label_size: int = 12,
                  legend_font_size: int = 12, annotaion_font_size: int = 12) -> None:

    color = call_color()
    ax = plt.subplot(1, 1, 1)
    ax.plot(np.arange(0, 101, 5), np.arange(0, 101, 5), color='black', label='Random', linewidth = line_size)
    # test color set
    if len(color) < len(var_list):
        for i in np.arange(len(color), len(var_list)):
            color.append(np.random.rand(3, 1))
    if len(label) < len(var_list):
        for i in np.arange(len(label), len(var_list)):
            label.append('line %s' % var_list[i])
    arrow_x_list = []
    for i, var in enumerate(var_list):
        # get pivot data
        t = pivot_num(data, var, performance=performance, n=10, ks=True, max_ks_only=False)
        t = t.loc[:9, ['cum_bad', 'cum_good', 'ks']]
        t0 = pd.DataFrame({'cum_bad': [0, ], 'cum_good': [0, ], 'ks': ['0%', ]}, index=[0, ])
        t = pd.concat([t0, t]).reset_index()
        t['ks'] = [float(str(x)[:-1]) for x in t.ks]
        point_index = np.argmax(t['ks'])
        arrow_x = t.loc[point_index, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100
        arrow_y = t.loc[point_index, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100

        if arrow_y > arrow_x:
            ax.plot(t.loc[:10, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100,
                    t.loc[:10, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100,
                    color=color[i], label=label[i], linewidth=line_size)
            ax.set_xlabel('%cum good', fontsize=x_label_size)
            ax.set_ylabel('%cum bad', fontsize=y_label_size)
        else:
            arrow_x, arrow_y = arrow_y, arrow_x
            ax.plot(t.loc[:10, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100,
                    t.loc[:10, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100,
                    color=color[i], label=label[i], linewidth=line_size)
            ax.set_ylabel('%cum good', fontsize=y_label_size)
            ax.set_xlabel('%cum bad', fontsize=x_label_size)
        j = 0
        while len(arrow_x_list) > 0 and min(abs(arrow_x_list - arrow_x - 3*(j+1))) < 5:
            j += 1
        ax.annotate(s='', xy=(arrow_x, arrow_x), xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color=color[i]))
        ax.annotate('%s ks = %.1f%%'%(label[i], t['ks'].max()),
                    xy=(arrow_x, arrow_x - 3*(j+1)), xytext=(5, 0),
                    textcoords='offset points', color=color[i], fontsize=annotaion_font_size)
        arrow_x_list.append(arrow_x)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f%%'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f%%'))
    ax.legend(loc='upper left', prop={'size': legend_font_size})
    plt.title(title, fontsize=title_font_size)
    plt.show()


def group_scatter_plot(df: PandasDF, group_var: str, x: str,
                       y: str, x_group_n: int = 10, fill_std: bool = False,
                       ticks_font: int = 10, label_font: int = 14) -> None:
    color = call_color()
    ax = plt.subplot(1, 1, 1)
    temp = df[group_var + [x, y]].copy()
    # xlim_low, xlim_high =
    # np.nanmedian(temp[x]) - 3 * np.nanstd(temp[x]), np.nanmedian(temp[x]) + 3 * np.nanstd(temp[x])
    temp = temp.loc[[str(a) != 'nan' and str(b) != 'nan' for
                     a, b in zip(temp[x], temp[y])], :]
    temp['x_q'] = pd.qcut(temp[x], n=x_group_n, format='order')
    x_ticks = temp[x].tolist()
    x_ticks.sort()
    x_index = [len(x_ticks)/x_group_n * i for i in range(1, x_group_n + 1)]
    x_ticks = [x_ticks[math.floor(i)-1] for i in x_index]
    try:
        x_ticks = [float(i) for i in x_ticks]
        if np.mean(x_ticks) < 1:
            x_ticks = ['<= %.1f%%' % (i*100) for i in x_ticks]
        else:
            x_ticks = ['<= %.1f' % i for i in x_ticks]
    except ValueError:
        x_ticks = ['<=%s' % i for i in x_ticks]
    for i, [key, df] in enumerate(temp.groupby(group_var)):
        df_dict = dict()
        for j in range(x_group_n):
            df['temp'] = 0
            df_dict[j] = df.loc[df.x_q <= j, :].groupby('temp').apply(lambda obj: pd.Series({
                'x_q': j,
                'y_avg': np.nanmean(obj[y]),
                'y_upper': np.nanmean(obj[y]) + np.nanstd(obj[y]),
                'y_lower': np.nanmean(obj[y]) - np.nanstd(obj[y])
                })).reset_index(drop=True)
        df = pd.concat([d for key, d in df_dict.items()]).reset_index(drop=True)
        if len(df) > 0:
            ax.plot(df['x_q'], df['y_avg'],
                    color=color[i], label=str(key), linewidth=4)
            if fill_std:
                ax.fill_between(df['x_q'], df['y_upper'], df['y_lower'], alpha=.2, color=color[i])
    ax.set_xlabel(x, fontsize=label_font)
    ax.set_ylabel(y, fontsize=label_font)
#     plt.xticks(range(len(x_ticks)), x_ticks, size='small')
    # Set number of ticks for x-axis
    ax.set_xticks(range(len(x_ticks)))
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks, fontsize=ticks_font)
    ax.tick_params(axis='both', which='major', labelsize=ticks_font)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=15)
    plt.show()


# descrete variables
def plot_des(df: PandasDF, var: str) -> None:
    temp = df[[var, 'BAD_IND_90DAYS', 'net_margin_90days_interest60delin', 'year_month']].copy()

    temp = temp.groupby(['year_month', var]).apply(lambda obj: pd.Series({
                'bad_rate': obj.BAD_IND_90DAYS.mean(),
                'net_margin': obj.net_margin_90days_interest60delin.mean(),
                'percentage': len(obj) / len(temp.loc[temp.year_month.isin(obj.year_month.unique()), :])})).reset_index()

    f, ax = plt.subplots(2, sharex=True)
    for i in temp[var].unique():
        temp_group = temp.loc[temp[var] == i, :]
        temp_group.sort_values('year_month', inplace = True)
        temp_group['year_month_index'] = np.arange(len(temp_group))
        ax[0].scatter(temp_group['year_month_index'], temp_group['bad_rate'], s=temp_group['percentage']*100,
                      label=None)
        ax[0].plot(temp_group['year_month_index'], temp_group['bad_rate'], label=i)
        ax[1].scatter(temp_group['year_month_index'], temp_group['net_margin'], s=temp_group['percentage']*100,
                      label=None)
        ax[1].plot(temp_group['year_month_index'], temp_group['net_margin'], label=i)
    ax[1].set_xlabel(var)
    ax[0].set_ylabel('bad rate')
    ax[1].set_ylabel('net margin')

    ax[0].set_xticks(range(len(temp['year_month'].unique())))
    xticks = temp.year_month.unique()
    xticks.sort()
    ax[0].set_xticklabels(xticks)
    plt.legend()
    plt.show()


def plot_con(df: PandasDF, var: str) -> None:
    temp = df[[var, 'BAD_IND_90DAYS', 'net_margin_90days_interest60delin', 'year_month']].copy()
    temp[var] = pd.qcut(temp[var], n=5)
    temp = temp.groupby(['year_month', var]).apply(lambda obj: pd.Series({
                'bad_rate': obj.BAD_IND_90DAYS.mean(),
                'net_margin': obj.net_margin_90days_interest60delin.mean(),
                'percentage': len(obj) / len(temp.loc[temp.year_month.isin(obj.year_month.unique()), :])})).reset_index()
    # plot
    f, ax = plt.subplots(2, sharex=True)
    for i in temp[var].unique():
        temp_group = temp.loc[temp[var] == i, :]
        temp_group.sort_values('year_month', inplace=True)
        temp_group['year_month_index'] = np.arange(len(temp_group))
        ax[0].scatter(temp_group.year_month_index, temp_group.bad_rate, s=temp_group.percentage*100, label=None)
        ax[1].scatter(temp_group.year_month_index, temp_group.net_margin, s=temp_group.percentage*100, label=None)
        ax[0].plot(temp_group.year_month_index, temp_group.bad_rate, label=i)
        ax[1].plot(temp_group.year_month_index, temp_group.net_margin, label=i)
    ax[1].set_xlabel('year_month')
    ax[0].set_ylabel('bad rate')
    ax[0].set_xticks(range(len(temp['year_month'].unique())))
    xticks = temp.year_month.unique()
    xticks.sort()
    ax[0].set_xticklabels(xticks)
    plt.legend()
    plt.show()


def call_color() -> List[Union[str, List[float]]]:
    return ['#00BFFF', (153/255, 63/255, 0/255), (0/255, 117/255, 220/255),
            (76/255, 0/255, 92/255), (240/255, 163/255, 255/255), (25/255, 25/255, 25/255),
            (0/255, 92/255, 49/255), (43/255, 206/255, 72/255), (255/255, 204/255, 153/255),
            (128/255, 128/255, 128/255), (148/255, 255/255, 181/255), (143/255, 124/255, 0/255),
            (157/255, 204/255, 0/255), (194/255, 0/255, 136/255), (0/255, 51/255, 128/255),
            (255/255, 164/255, 5/255), (255/255, 168/255, 187/255), (66/255, 102/255, 0/255),
            (255/255, 0/255, 16/255), (94/255, 241/255, 242/255), (0/255, 153/255, 143/255),
            (224/255, 255/255, 102/255), (116/255, 10/255, 255/255), (153/255, 0/255, 0/255),
            (255/255, 255/255, 128/255), (255/255, 255/255, 0/255), (255/255, 80/255, 5)]
