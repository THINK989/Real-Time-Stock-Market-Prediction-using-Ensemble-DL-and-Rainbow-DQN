"""
Model and Forecast function
The LSTM,GRU model and forecast function implementation.
Copyright (c) 2018 HUSEIN ZOLKEPLI(huseinzol05)
Licensed under the Apache License Version 2.0 (see LICENSE for details)
Written by HUSEIN ZOLKEPLI
"""

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import timedelta
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
	minmax_for = MinMaxScaler().fit(data.iloc[:,[3,4,6]].astype('float32')) # Close index, Volume and Sentiment
	df_log = minmax_for.transform(data.iloc[:,[3,4,6]].astype('float32')) # Close index, Volume and Sentiment
	df_log = pd.DataFrame(df_log)
	return df_log,minmax_for

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

def forecast_LSTM(df_train, minmax_for, data):

	tf.compat.v1.reset_default_graph()
	modelnn = Model_LSTM(
		learning_rate, num_layers, df_train.shape[1], size_layer, df_train.shape[1], dropout_rate
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

def forecast_GRU(df_train, minmax_for,data):
    tf.compat.v1.reset_default_graph()
    modelnn = Model_GRU(
        learning_rate, num_layers, df_train.shape[1], size_layer, df_train.shape[1], dropout_rate
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