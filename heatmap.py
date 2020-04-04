import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import pandas as pd
import bs4 as bs
import pickle
import requests
import os
import pandas_datareader.data as web
import yfinance as yf

style.use('ggplot')

# Extract Current top 50 Stocks which are part of Nifty50
def save_nifty50_tickers():

	resp = requests.get('https://en.wikipedia.org/wiki/NIFTY_50')
	soup  = bs.BeautifulSoup(resp.text,'lxml')
	table  = soup.find('table', {'id':'constituents'})
	tickers  = []
	for row in table.findAll('tr')[1:]:
		ticker = row.findAll('td')[1].text
		tickers.append(ticker)
	with open('NIFTY_50.pickle','wb') as f:
		pickle.dump(tickers, f)

	#print(tickers)

	return tickers
#Tickers:- Conatains all the symbols of the stocks

#Get 1 min data for each symbol within a period of 7days
def get_data_from_yahoo(reload_nifty50 = False):
	if reload_nifty50:
		tickers = save_nifty50_tickers()
	else:
		with open('NIFTY_50.pickle', 'rb') as f:
			tickers = pickle.load(f)

	if not os.path.exists('stock_dfs'):
		os.makedirs('stock_dfs')	
				
	# start = dt.datetime(2010,1,1)
	# end = dt.datetime.now()

	for ticker in tickers:
		
		# if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
		df = yf.download(ticker, period = '7d',interval = '1m')
		df.to_csv('stock_dfs/{}.csv'.format(ticker.replace('\n','')))
		# else:
		# 	print('Already have {}'.format(ticker))


# Join the Adj Close columns of all the csv file into single dataframe replace the name with ticker
def compile_data():
	with open('NIFTY_50.pickle', 'rb') as f:
		tickers = pickle.load(f)
	main_df = pd.DataFrame()
	get_data_from_yahoo()
	for count, ticker in enumerate(tickers):
		df = pd.read_csv('stock_dfs/{}.csv'.format(ticker.replace('\n','')))
		df.set_index('Datetime', inplace=True)

		df.rename(columns={'Adj Close': ticker.replace('\n','')}, inplace=True)
		df.drop(['Open','High','Low','Close','Volume'], 1, inplace = True)

		if main_df.empty:
			main_df = df

		else:
			main_df = main_df.join(df, how='outer')

	#print(main_df.head())
	main_df.to_csv('nifty50_joined.csv')
	

# Vizualising the corelation on returns of the stocks to see if they follow normal distribution
def visualize_data():
	compile_data()
	df = pd.read_csv('nifty50_joined.csv')
	df.set_index('Datetime', inplace = True)

	df_corr = df.pct_change().corr()

	data = df_corr.values
	fig = plt.figure(figsize = (15, 10))
	ax = fig.add_subplot(1,1,1)
	heatmap = ax.pcolor(data, cmap = plt.cm.RdYlBu)
	fig.colorbar(heatmap)

	ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
	ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

	ax.invert_yaxis()
	ax.xaxis.tick_top()
	column_labels = df_corr.columns
	row_labels = df_corr.index

	ax.set_xticklabels(column_labels)
	ax.set_yticklabels(row_labels)

	plt.xticks(rotation = 90)
	heatmap.set_clim(-1,1)
	plt.tight_layout()
	fig.savefig('heatmap.png')

visualize_data()



