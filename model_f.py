
# coding: utf-8
# update: 10/27/2017


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import math 

pd.options.mode.chained_assignment = None
	
def pred_q(var_list, n = 10, format = 'origin'):
	"""
	Use pd.qcut instead.
	This function produce ``var``n-th quantile based on given ``n``
	Parameter:
		var_list(list): a list to get q
		n(int): n-th quantile
		format: ['percentage', 'origin', 'order']
	Return:
		data(pandas dataframe): new data frame with ``'%s_q'%var``
		"""
	# pred_quantile
	col = np.array(var_list)
	q = np.full_like(col, np.nan)
	q = q.astype('object')
	points = [np.nanpercentile(col, 100/n*i) for i in range(n+1)]
	for i in range(n):
		level = i
		if format == 'origin':
			level = '%s to %s'%(points[i], points[i+1])
		else:
			level = '%.1f%% to %.1f%%'%(points[i], points[i+1])
		q[np.array([points[i] <= x <= points[i+1] for x in col])] = level
	
	return q


def pred_range(var_list, range_l = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], format = 'origin'):
	"""
	use pd.cut instead.
	This function produce ``var``segment based on given ``range_l``
	Parameter:
		df(pandas dataframe): data
		var(string): column name
		range_l(list): list of range cut point
		format: ['percentage', 'origin', 'order']
	Return:
		data(pandas dataframe): new data frame with ``'%s_range'%var``
	"""
	col = np.array(var_list)
	col_range = np.full_like(col, '-1')
	col_range = col_range.astype('object')
	for i in range(len(range_l)-1):
		lower = range_l[i]
		upper = range_l[i+1]
		if format == 'order':
			level = i
		elif format == 'percentage':
			level = "%.1f%%" % (lower * 100) + ' - ' + "%.1f%%" % (upper * 100)
		elif format == 'int':
			level = '%d - %d'%(lower, upper)
		else:
			level = '%.1f - %.1f'%(lower, upper)
		col_range[np.array([lower <= x < upper for x in col.flat])]= level
		
	return col_range

def pivot_num(data, var, performance = 'bad_ind', n = 10, ks = True, max_ks_only = False):
	""" Output the bad rate segment for a particular variable in group
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
	temp_noMissing.sort_values(var, inplace = True)
	length = round(temp_noMissing.shape[0]/n)
	
	pivot = lambda obj: pd.Series({
			'var': var,
			'level': str(obj[var].min()) + ' - ' +  str(obj[var].max()),
			'bad rate': obj[performance].mean(), 
			'count': len(obj[performance])
		})
	group = temp_noMissing.groupby(np.arange(temp_noMissing.shape[0]) // length).apply(pivot)
	group_missing = pd.DataFrame({
			'var': var,
			'level': np.nan,
			'bad rate': temp_missing[performance].mean(),
			'count': temp_missing.shape[0],
			'ks': np.nan
		}, index = [n+1,])
	# temp = group[['bad rate', 'count']].copy()
	if ks or max_ks_only:
		group['bad'] = [r * c for r,c in zip(group['bad rate'], group['count'])]
		group['cum_bad'] = [sum(group.loc[0:i, 'bad']) for i in range(group.shape[0])]
		group['cum_count'] = [sum(group.loc[0:i, 'count']) for i in range(group.shape[0])]
		group['cum_good'] = [c - b for c, b in zip(group['cum_count'], group['cum_bad'])]
		group['ks'] = [(100 * abs(g/group.loc[group.shape[0]-1,'cum_good'] - b/group.loc[group.shape[0]-1, 'cum_bad'])) \
		for g, b in zip(group.cum_good, group.cum_bad)]
		max_index = group['ks'].idxmax()
		if max_ks_only:
			return group.loc[[max_index,], ['var', 'ks']]
		group['ks'] = ['%.1f%%'%x for x in group['ks']]
		group = group.append(group_missing)
		group['bad rate'] = ['%.2f%%'%(x * 100) for x in group['bad rate']]
		
		def highlight(s):
			return 'background-color: yellow'
		group.style.applymap(highlight, subset = pd.IndexSlice[max_index, ['ks']])
		
		return group[['var', 'level', 'bad rate', 'count', 'cum_bad', 'cum_good', 'ks']]
	else:
		group = group.append(group_missing[['var', 'level', 'bad rate', 'count']])
		group.rename(columns = {'bad rate': 'avg %s'%performance}, inplace = True)
		return group[['var', 'level', 'avg %s'%performance, 'count']]


	

def model_ks_plot(data, var_list, label = ['Risk Model', ], performance = 'bad_ind',
	title = 'Model Performance', 
	line_size = 1,
	title_font_size = 18,
	x_label_size = 12, y_label_size = 12,
	legend_font_size = 12, annotaion_font_size = 12):

	color = call_color()
	ax = plt.subplot(1,1,1)
	ax.plot(np.arange(0,101,5), np.arange(0,101, 5), color = 'black', label = 'Random', linewidth = line_size)
	# test color set
	if len(color) < len(var_list):
		for i in np.arange(len(color), len(var_list)):
			color.append(np.random.rand(3,1))
	if len(label) < len(var_list):
		for i in np.arange(len(label), len(var_list)):
			label.append('line %s'%var_list[i])
	arrow_x_list = []
	for i,var in enumerate(var_list):
		# get pivot data
		t = pivot_num(data, var, performance = performance, n = 10, ks = True, max_ks_only = False)
		t = t.loc[:9, ['cum_bad', 'cum_good', 'ks']]
		t0 = pd.DataFrame({'cum_bad': [0,], 'cum_good': [0,], 'ks': ['0%',]}, index = [0,])
		t = pd.concat([t0, t]).reset_index()
		t['ks'] = [float(str(x)[:-1]) for x in t.ks]
		point_index = np.argmax(t['ks'])
		arrow_x = t.loc[point_index, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100
		arrow_y = t.loc[point_index, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100
		
		if arrow_y > arrow_x: 
			ax.plot(t.loc[:10, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100,
					t.loc[:10, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100, 
					color = color[i], label = label[i], linewidth = line_size)
			ax.set_xlabel('%cum good', fontsize = x_label_size)
			ax.set_ylabel('%cum bad', fontsize = y_label_size)
		else:
			arrow_x, arrow_y = arrow_y, arrow_x
			ax.plot(t.loc[:10, 'cum_bad']/t.loc[10, 'cum_bad'].sum() * 100,
					t.loc[:10, 'cum_good']/t.loc[10, 'cum_good'].sum() * 100,
					color = color[i], label = label[i], linewidth = line_size)
			ax.set_ylabel('%cum good', fontsize = y_label_size)
			ax.set_xlabel('%cum bad', fontsize = x_label_size)
		j = 0
		while len(arrow_x_list) > 0 and min(abs(arrow_x_list - arrow_x - 3*(j+1))) < 5:
			j += 1 
		ax.annotate(s='', xy=(arrow_x, arrow_x), xytext=(arrow_x,arrow_y), 
			arrowprops=dict(arrowstyle='<->', color = color[i])) 
		ax.annotate('%s ks = %.1f%%'%(label[i],t['ks'].max()), 
			xy=(arrow_x, arrow_x - 3*(j+1)), xytext=(5,0),
			 textcoords='offset points', color = color[i], fontsize = annotaion_font_size) 
		arrow_x_list.append(arrow_x)
		
	
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f%%'))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f%%'))
	ax.legend(loc='upper left', prop = {'size': legend_font_size})
	plt.title(title, fontsize = title_font_size)
	plt.show()
	return ax

# def profit_summary(data, var):
# 	"""This function summarize the profit based on ``var`` and its grouping ``var_grouping``.
# 	@param:
# 		data(pandas dataframe): pandas dataframe
# 		var(list): list of var for grouping
# 	@return:
# 		df_result(pandas dataframe): a dataframe of profit summary.
# 	"""
# 	if isinstance(var, (list, tuple)):
# 		df = data[var + [ 'net_margin_90days', 'net_margin_180days', 'FIRST_PRINCIPAL', 
# 				   'BAD_IND_90DAYS', 'BAD_IND_180DAYS', 
# 				  ]]
# 	else:
# 		df = data[[var, 'net_margin_90days', 'net_margin_180days', 'FIRST_PRINCIPAL', 
# 				   'BAD_IND_90DAYS', 'BAD_IND_180DAYS', 
# 				  ]]
# 	# function for profit summary
# 	fn = lambda obj: pd.Series({'num of loans': len(obj), 
# 							   'loan amt': np.nansum(obj.FIRST_PRINCIPAL),
# 							   'loan percentage': np.nansum(obj.FIRST_PRINCIPAL)/np.nansum(df.FIRST_PRINCIPAL),
# 							   'avg loan amt': np.nanmean(obj.FIRST_PRINCIPAL),
# 							   'bad rate 90days': np.nanmean(obj.BAD_IND_90DAYS),
# 							   'bad rate 180days': np.nanmean(obj.BAD_IND_180DAYS),
# 							   'net margin 90days': np.nansum(obj.net_margin_90days),
# 							   'net margin 90days distr': np.nansum(obj.net_margin_90days)/np.nansum(df.net_margin_90days),
# 								'net margin 90days per loan': np.nanmean(obj.net_margin_90days),
# 							   'net margin 180days': np.nansum(obj.net_margin_180days),
# 							   'net margin 180days distr': np.nansum(obj.net_margin_180days)/np.nansum(df.net_margin_180days),
# 							   'net margin 180days per loan': np.nanmean(obj.net_margin_180days),
# 							   })
# 	df['total'] = 'Total'

	
# 	df0 = df.groupby(var).apply(fn).reset_index()
# 	df1 = df.groupby('total').apply(fn).reset_index(drop = True)
# 	if isinstance(var, (list, tuple)):
# 		for i in var:
# 			df1[i] = 'total'
# 	else:
# 		df[var] = 'total'
# 	df = pd.concat([df0, df1]).reset_index(drop = True)
# 	# format before output
# 	for i in ['avg loan amt', 'loan amt', 'net margin 180days', 'net margin 90days', 
# 			  'net margin 90days per loan', 'net margin 180days per loan',
# 			  'net margin i 180days', 'net margin i 180days per loan']:  
# 		df[i] =  ["${:,.1f}".format(x) for x in df[i]]
# 	for i in ['bad rate 180days', 'bad rate 90days', 'loan percentage', 'net margin 180days distr', 'net margin 90days distr', ]:
# 		df[i] = ['%.1f%%'%(100*x) for x in df[i]]
# 	df['num of loans'] = ['%d'%x for x in df['num of loans']]
# 	return df


def group_scatter_plot(df, group_var, x, y, x_group_n = 10, fill_std = False, ticks_font = 10, label_font = 14):
	color = call_color()
	ax = plt.subplot(1,1,1)
	temp = df[group_var + [x, y]].copy()
	# xlim_low, xlim_high = np.nanmedian(temp[x]) - 3 * np.nanstd(temp[x]), np.nanmedian(temp[x]) + 3 * np.nanstd(temp[x])
	temp = temp.loc[[str(a) != 'nan' and str(b) != 'nan' for 
					 a,b in zip(temp[x], temp[y])], :]
	temp['x_q'] = pred_q(temp[x], n = x_group_n, format = 'order')
	x_ticks = temp[x].tolist()
	x_ticks.sort()
	x_index = [len(x_ticks)/x_group_n * i for i in range(1, x_group_n + 1)]
	x_ticks = [x_ticks[math.floor(i)-1] for i in x_index]
	try:
		x_ticks = [float(i) for i in x_ticks]
		if np.mean(x_ticks) < 1:
			x_ticks = ['<= %.1f%%'%(i*100) for i in x_ticks]
		else:
			x_ticks = ['<= %.1f'%i for i in x_ticks]
	except:
		x_ticks = ['<=%s'%i for i in x_ticks]
	for i, [key, df] in enumerate(temp.groupby(group_var)):
		df_dict = dict()
		for j in range(x_group_n):
			df['temp'] = 0
			df_dict[j] = df.loc[df.x_q <= j, :].groupby('temp').apply(lambda obj: pd.Series({
				'x_q': j,
				'y_avg': np.nanmean(obj[y]),
				'y_upper': np.nanmean(obj[y]) + np.nanstd(obj[y]),
				'y_lower': np.nanmean(obj[y]) - np.nanstd(obj[y])
				})).reset_index(drop = True)
		df = pd.concat([d for key, d in df_dict.items()]).reset_index(drop = True)
		if len(df) > 0:
			ax.plot(df['x_q'], df['y_avg'], 
					color = color[i], label = str(key), linewidth = 4)
			if fill_std:
				ax.fill_between(df['x_q'], df['y_upper'], df['y_lower'], alpha = .2, color = color[i])
	ax.set_xlabel(x, fontsize = label_font)
	ax.set_ylabel(y, fontsize = label_font)
#     plt.xticks(range(len(x_ticks)), x_ticks, size='small')
	# Set number of ticks for x-axis
	ax.set_xticks(range(len(x_ticks)))
	# Set ticks labels for x-axis
	ax.set_xticklabels(x_ticks, fontsize=ticks_font)
	ax.tick_params(axis = 'both', which = 'major', labelsize=ticks_font)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
		  ncol=3, fancybox=True, shadow=True)
	labels = ax.get_xticklabels()
	plt.setp(labels, rotation=15)
	plt.show()


# descrete variables
def plot_des(df, var):
	temp = df[[var, 'BAD_IND_90DAYS', 'net_margin_90days_interest60delin', 'year_month']].copy()

	temp = temp.groupby(['year_month',var]).apply(lambda obj: pd.Series({
				'bad_rate': obj.BAD_IND_90DAYS.mean(),
				'net_margin': obj.net_margin_90days_interest60delin.mean(),
				'percentage': len(obj) / len(temp.loc[temp.year_month.isin(obj.year_month.unique()), :])})).reset_index()

	f, ax = plt.subplots(2, sharex = True)
	for i in temp[var].unique():
		temp_group = temp.loc[temp[var] == i, :]
		temp_group.sort_values('year_month', inplace = True)
		temp_group['year_month_index'] = np.arange(len(temp_group))
		ax[0].scatter(temp_group['year_month_index'], temp_group['bad_rate'], s = temp_group['percentage']*100, label = None)
		ax[0].plot(temp_group['year_month_index'], temp_group['bad_rate'], label = i)
		ax[1].scatter(temp_group['year_month_index'], temp_group['net_margin'], s = temp_group['percentage']*100, label = None)
		ax[1].plot(temp_group['year_month_index'], temp_group['net_margin'], label = i)
	ax[1].set_xlabel(var)
	ax[0].set_ylabel('bad rate')
	ax[1].set_ylabel('net margin')

	ax[0].set_xticks(range(len(temp['year_month'].unique())))
	xticks = temp.year_month.unique()
	xticks.sort()
	ax[0].set_xticklabels(xticks)
	plt.legend()
	plt.show()

def plot_con(df, var):
	temp = df[[var, 'BAD_IND_90DAYS', 'net_margin_90days_interest60delin', 'year_month']].copy()
	temp[var] = pred_q(temp[var], n=5)
	temp = temp.groupby(['year_month',var]).apply(lambda obj: pd.Series({
				'bad_rate': obj.BAD_IND_90DAYS.mean(),
				'net_margin': obj.net_margin_90days_interest60delin.mean(),
				'percentage': len(obj) / len(temp.loc[temp.year_month.isin(obj.year_month.unique()), :])})).reset_index()
	# plot
	f, ax = plt.subplots(2,sharex = True)
	for i in temp[var].unique():
		temp_group = temp.loc[temp[var] == i, :]
		temp_group.sort_values('year_month', inplace = True)
		temp_group['year_month_index'] = np.arange(len(temp_group))
		ax[0].scatter(temp_group.year_month_index, temp_group.bad_rate, s = temp_group.percentage*100, label = None)
		ax[1].scatter(temp_group.year_month_index, temp_group.net_margin, s = temp_group.percentage*100, label = None)
		ax[0].plot(temp_group.year_month_index, temp_group.bad_rate, label = i)
		ax[1].plot(temp_group.year_month_index, temp_group.net_margin, label = i)
	ax[1].set_xlabel('year_month')
	ax[0].set_ylabel('bad rate')
	ax[0].set_xticks(range(len(temp['year_month'].unique())))
	xticks = temp.year_month.unique()
	xticks.sort()
	ax[0].set_xticklabels(xticks)
	plt.legend()
	plt.show()




def call_color():
	return ['#00BFFF',(153/255,63/255,0/255),(0/255,117/255,220/255),
	(76/255,0/255,92/255),(240/255,163/255,255/255),(25/255,25/255,25/255),
	(0/255,92/255,49/255),(43/255,206/255,72/255),(255/255,204/255,153/255),
	(128/255,128/255,128/255),(148/255,255/255,181/255),(143/255,124/255,0/255),
	(157/255,204/255,0/255),(194/255,0/255,136/255),(0/255,51/255,128/255),
	(255/255,164/255,5/255),(255/255,168/255,187/255),(66/255,102/255,0/255),
	(255/255,0/255,16/255),(94/255,241/255,242/255),(0/255,153/255,143/255),
	(224/255,255/255,102/255),(116/255,10/255,255/255),(153/255,0/255,0/255),
	(255/255,255/255,128/255),(255/255,255/255,0/255),(255/255,80/255,5)]