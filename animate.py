"""
Model and Forecast function
The LSTM,GRU model and forecast function implementation.
Copyright (c) 2018 HUSEIN ZOLKEPLI(huseinzol05)
Licensed under the Apache License Version 2.0 (see LICENSE for details)
Written by HUSEIN ZOLKEPLI
"""
import sys
sys.path.insert(0,'rainbow/')
import warnings
warnings.filterwarnings('ignore')
import datetime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import tensorflow as tf
import matplotlib
import pickle 
import pylab
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tqdm import tqdm
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from matplotlib import style
from eval import main
from matplotlib import animation
from sentiment import news2sentiment



style.use('fivethirtyeight')

matplotlib.rcParams.update({'font.size': 9})

# Go to https://www.alphavantage.co/support/#api-key
# Generate the key 1TYVS0JD8DTD2H9O
#Put your key in key parameter
ts = TimeSeries(key='96FB5MR6YB0T07OU',output_format='pandas')
count = -1



#################################################################################


#############                       FORECAST                           #########

## Section adopted from https://github.com/huseinzol05/Stock-Prediction-Models ##
#################################################################################

"""
Made changes in 

Variable: - hyper-parameter tuning
Class Model_LSTM: - Remove Deprication Warnings  
Class Model_GRU: - Remove Deprication Warnings  
def forecast_LSTM(): - Remove Deprication Warnings 
def forecast_GRU(): - Remove Deprication Warnings 
Added 2 more parameters ['Volume','Sentiment'] to the networks 
"""
simulation_size = 1
num_layers = 1
size_layer = 128
timestamp = 4
epoch = 69 
dropout_rate = 0.8
test_size = 10
learning_rate = 0.01


def preprocess_data(data, scores):
	df_score = pd.DataFrame(scores)
	data = data.join(df_score.set_index(data.index))
	global minmax_for
	minmax_for = MinMaxScaler().fit(data.iloc[:,[3,4,6]].astype('float32')) # Close index, Volume and Sentiment
	global df_log
	df_log = minmax_for.transform(data.iloc[:,[3,4,6]].astype('float32')) # Close index, Volume and Sentiment
	df_log = pd.DataFrame(df_log)
	return df_log

class Model_LSTM:
	def __init__(
		self,
		learning_rate,
		num_layers,
		size,
		size_layer,
		output_size,
		forget_bias = 0.1,
	):
		def lstm_cell(size_layer):
			return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

		rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
			[lstm_cell(size_layer) for _ in range(num_layers)],
			state_is_tuple = False,
		)
		self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
		self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
		drop = tf.contrib.rnn.DropoutWrapper(
			rnn_cells, output_keep_prob = forget_bias
		)
		self.hidden_layer = tf.compat.v1.placeholder(
			tf.float32, (None, num_layers * 2 * size_layer)
		)
		self.outputs, self.last_state = tf.nn.dynamic_rnn(
			drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
		)
		self.logits = tf.layers.dense(self.outputs[-1], output_size)
		self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
			self.cost
		)
class Model_GRU:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.GRUCell(size_layer)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
        self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.compat.v1.placeholder(
            tf.float32, (None, num_layers * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        

	
def calculate_accuracy(real, predict):
	real = np.array(real) + 1
	predict = np.array(predict) + 1
	percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
	return percentage * 100

def anchor(signal, weight):
	buffer = []
	last = signal[0]
	for i in signal:
		smoothed_val = last * weight + (1 - weight) * i
		buffer.append(smoothed_val)
		last = smoothed_val
	return buffer

def forecast_LSTM():

	tf.compat.v1.reset_default_graph()
	modelnn = Model_LSTM(
		learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
	)
	sess = tf.compat.v1.InteractiveSession()
	sess.run(tf.compat.v1.global_variables_initializer())
	date_ori = pd.to_datetime(data.index).tolist()

	pbar = tqdm(range(epoch), desc = 'LSTM train loop')
	for i in pbar:
		init_value = np.zeros((1, num_layers * 2 * size_layer))
		total_loss, total_acc = [], []
		for k in range(0, df_train.shape[0] - 1, timestamp):
			index = min(k + timestamp, df_train.shape[0] - 1)
			batch_x = np.expand_dims(
				df_train.iloc[k : index, :].values, axis = 0
			)
			# print('batch_x: -', batch_x)
			batch_y = df_train.iloc[k + 1 : index + 1, :].values
			# print('batch_y: -', batch_y)
			logits, last_state, _, loss = sess.run(
				[modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
				feed_dict = {
					modelnn.X: batch_x,
					modelnn.Y: batch_y,
					modelnn.hidden_layer: init_value,
				},
			)        
			init_value = last_state
			total_loss.append(loss)
			total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
		pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
	
	future_day = test_size

	output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
	output_predict[0] = df_train.iloc[0]
	upper_b = (df_train.shape[0] // timestamp) * timestamp
	init_value = np.zeros((1, num_layers * 2 * size_layer))

	for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
		out_logits, last_state = sess.run(
			[modelnn.logits, modelnn.last_state],
			feed_dict = {
				modelnn.X: np.expand_dims(
					df_train.iloc[k : k + timestamp], axis = 0
				),
				modelnn.hidden_layer: init_value,
			},
		)
		init_value = last_state
		output_predict[k + 1 : k + timestamp + 1] = out_logits

	if upper_b != df_train.shape[0]:
		out_logits, last_state = sess.run(
			[modelnn.logits, modelnn.last_state],
			feed_dict = {
				modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
				modelnn.hidden_layer: init_value,
			},
		)
		output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
		future_day -= 1
		date_ori.append(date_ori[-1] + timedelta(days = 1))

	init_value = last_state
	
	for i in range(future_day):
		o = output_predict[-future_day - timestamp + i:-future_day + i]
		out_logits, last_state = sess.run(
			[modelnn.logits, modelnn.last_state],
			feed_dict = {
				modelnn.X: np.expand_dims(o, axis = 0),
				modelnn.hidden_layer: init_value,
			},
		)
		init_value = last_state
		output_predict[-future_day + i] = out_logits[-1]
		date_ori.append(date_ori[-1] + timedelta(days = 1))
	
	output_predict = minmax_for.inverse_transform(output_predict)
	deep_future = anchor(output_predict[:, 0], 0.4)

	
	return deep_future

def forecast_GRU():
    tf.compat.v1.reset_default_graph()
    modelnn = Model_GRU(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    date_ori = pd.to_datetime(data.index).tolist()

    pbar = tqdm(range(epoch), desc = 'GRU train loop')
    for i in pbar:
        init_value = np.zeros((1, num_layers * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k : index, :].values, axis = 0
            )
            batch_y = df_train.iloc[k + 1 : index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict = {
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )        
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
    
    future_day = test_size

    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    df_train.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days = 1))

    init_value = last_state
    
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(o, axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    
    output_predict = minmax_for.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)
    
    return deep_future

###################################################

###               STOCK PLOTTING                ### 

###################################################
def rsiFunc(prices, n=14):
	deltas = np.diff(prices)
	seed = deltas[:n+1]
	up = seed[seed>=0].sum()/n
	down = -seed[seed<0].sum()/n
	rs = up/down
	rsi = np.zeros_like(prices)
	rsi[:n] = 100. - 100./(1.+rs)

	for i in range(n, len(prices)):
		delta = deltas[i-1] # cause the diff is 1 shorter

		if delta>0:
			upval = delta
			downval = 0.
		else:
			upval = 0.
			downval = -delta

		up = (up*(n-1) + upval)/n
		down = (down*(n-1) + downval)/n

		rs = up/down
		rsi[i] = 100. - 100./(1.+rs)

	return rsi

def movingaverage(values,window):
	weigths = np.repeat(1.0, window)/window
	smas = np.convolve(values, weigths, 'valid')
	return smas # as a numpy array


def ExpMovingAverage(values, window):
	weights = np.exp(np.linspace(-1., 0., window))
	weights /= weights.sum()
	a =  np.convolve(values, weights, mode='full')[:len(values)]
	a[:window] = a[window]
	return a


def computeMACD(x, slow=26, fast=12):
	"""
	compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(x) arrays
	"""
	emaslow = ExpMovingAverage(x, slow)
	emafast = ExpMovingAverage(x, fast)
	return emaslow, emafast, emafast - emaslow


def bytespdate2num(fmt, encoding='utf-8'):
	strconverter = mdates.strpdate2num(fmt)
	def bytesconverter(b):
		s = b.decode(encoding)
		return strconverter(s)
	return bytesconverter



def graphData(stock,MA1,MA2,interval):

	global data
	fig.clf()

	'''
		Use this to dynamically pull a stock:
	'''
	try:
		
		print('Currently Pulling',stock)
		data, meta_data = ts.get_intraday(symbol=stock,interval=str(interval)+'min')
		# data.to_csv('data/NSEI1min.csv')
		# data = data.iloc[::-1]
		data['date'] = data.index
		data['date'] = data['date'].map(mdates.date2num)

		#print(data)
		global df_train
		scores = news2sentiment()
		df_train = preprocess_data(data, scores)
		#print('Data Frame:-',df_train)
		
	except Exception as e:
		print(str(e), 'failed to pull pricing data')

	# try:
	## preparation for candlestick
	date, openp, highp, lowp, closep, volume = data['date'].tolist(), data['1. open'].tolist(), data['2. high'].tolist(), data['3. low'].tolist(), data['4. close'].tolist(), data['5. volume'].tolist()
	x = 0
	y = len(date)
	newAr = []
	while x < y:
		appendLine = date[x],openp[x],highp[x],lowp[x],closep[x],volume[x]
		newAr.append(appendLine) #contains data for candlestick ohlc plot
		x+=1


	global count
	global results_backup
	global results_backup_lstm
	global results_backup_gru
	global date2add
	global prev_date
	# main axis in the figure
	ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
	# candlestickohlc plot from mplfinanace
	candlestick_ohlc(ax1, newAr, width=.0005, colorup='#53c156')

	# workaround for updating plot after interval size without disturbing the forecast plot
	if count == -1:
		results = []
		results_LSTM = []
		results_GRU = []
		temp = []
		results_LSTM.append(forecast_LSTM())
		results_GRU.append(forecast_GRU())
		for i in range(len(results_LSTM[0])):
			temp.append((results_LSTM[0][i] + results_GRU[0][i])/2)
		results.append(temp)
		results_backup = results 
		# results_backup_lstm = results_LSTM
		# results_backup_gru = results_GRU
		manual_run = False
		main(temp, 10, "model_noisynstepperdddqn_20", True, manual_run)
		#print('Results Leng:-', len(results[0]))
		prev_date = date
		ax1.axvline(x=date[-1], color = 'r',linewidth=2)
		date2add = [date[-1]]
		# print(date)
		for i in range(test_size):
			date2add.append(date2add[-1] + (0.0006944444*interval))
		# for no, r in enumerate(results_LSTM):
		# 	ax1.plot(date + date2add[1:],r, label = 'LSTM', linewidth = 2, alpha = 0.5)
		# for no, r in enumerate(results_GRU):
		# 	ax1.plot(date + date2add[1:],r, label = 'GRU', linewidth = 2, alpha = 0.5)
		for no, r in enumerate(results):
			ax1.plot(date + date2add[1:],r, label = 'LSTM+GRU', linewidth = 2)

		count += 1
	else:
		if count == test_size-1:
			ax1.axvline(x=prev_date[-1], color = 'r',linewidth=2)
			# for no, r in enumerate(results_backup_lstm):
			# 	ax1.plot(date + date2add[1:],r, label = 'LSTM', linewidth = 2, alpha = 0.5)
			# for no, r in enumerate(results_backup_gru):
			# 	ax1.plot(date + date2add[1:],r, label = 'GRU', linewidth = 2, alpha = 0.5)
			for no, r in enumerate(results_backup):
				ax1.plot(prev_date + date2add[1:],r, label = 'LSTM+GRU', linewidth = 2)
			count = -1
		elif count != test_size:
			ax1.axvline(x=prev_date[-1], color = 'r',linewidth=2)
			# for no, r in enumerate(results_backup_lstm):
			# 	ax1.plot(date + date2add[1:],r, label = 'LSTM', linewidth = 2, alpha = 0.5)
			# for no, r in enumerate(results_backup_gru):
			# 	ax1.plot(date + date2add[1:],r, label = 'GRU', linewidth = 2, alpha = 0.5)
			for no, r in enumerate(results_backup):
				ax1.plot(prev_date + date2add[1:],r, label = 'LSTM+GRU', linewidth = 2)
			count += 1


	# Plotting close price on top of candlestick ohlc
	ax1.plot(date, closep, color = '#e75480', label = 'Closing Price', linewidth=2)
	
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(20))
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
	
	plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

	plt.ylabel('Stock price and Volume')

	maLeg = plt.legend(loc=9, ncol=2, prop={'size':10},
			   fancybox=True, borderaxespad=0.)
	maLeg.get_frame().set_alpha(0.4)
	textEd = pylab.gca().get_legend().get_texts()
	pylab.setp(textEd[0:5])

	volumeMin = 0
	ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4)
	rsi = rsiFunc(closep)
	posCol = '#386d13'
	negCol = '#8f2020'
	plt.title(stock.upper())
	ax0.plot(date, rsi, linewidth=1.5)
	ax0.axhline(70,color = negCol)
	ax0.axhline(30, color = posCol)
	ax0.fill_between(date, rsi, 70, where=(rsi>=70), alpha=0.5)
	ax0.fill_between(date, rsi, 30, where=(rsi<=30), alpha=0.5)
	ax0.set_yticks([30,70])
	plt.ylabel('RSI')

	ax1v = ax1.twinx()
	ax1v.fill_between(date,volumeMin, volume, facecolor = '#ffd700',alpha=.5)
	ax1v.axes.yaxis.set_ticklabels([])
	ax1v.grid(False)
	###Edit this to 3, so it's a bit larger
	ax1v.set_ylim(0, 3*max(volume))
	ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4)
	nslow = 26
	nfast = 12
	nema = 9
	emaslow, emafast, macd = computeMACD(closep)
	ema9 = ExpMovingAverage(macd, nema)
	ax2.plot(date, macd, lw=2)
	ax2.plot(date, ema9, lw=1)
	ax2.fill_between(date, macd-ema9, 0, alpha=0.5)

	plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
 
	plt.ylabel('MACD')
	ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
	plt.xticks(rotation = 45)

	
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	
	plt.tight_layout()
	datetimeobj = datetime.now()
	if interval == 1:
		print('Wait for '+str(interval)+' minute')
	else:
		print('Wait for '+str(interval)+' minutes')
	# plt.show()

	## To save every plot
	fig.savefig('test_image/example'+str(datetimeobj.hour)+'_'+str(datetimeobj.minute) +'_'+str(datetimeobj.second)+'.png',facecolor=fig.get_facecolor())
	   
	# except Exception as e:
	# 	print('main loop',str(e))

fig = plt.figure(figsize =(20,9))

def animate(i):
	graphData(stock,10,50,interval)

def interval_period():
	interval_check = int(input('Enter the amount of interval among 1, 5, 15 and 30:-'))
	return interval_check


while True:
	stock = input('Stock to plot: ')
	interval = interval_period() 
	validator = [1,5,15,30]
	while interval not in validator:
		print('Choose any one from the given options')
		interval = interval_period()

	ani = animation.FuncAnimation(fig, animate, interval = 60000*interval)
	ani.event_source.start()
	if interval == 1:
		print('Plot will update every '+str(interval)+' minute')
	else:
		print('Plot will update every '+str(interval)+' minutes')
	plt.show()