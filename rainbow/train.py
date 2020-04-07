
"""
Script for training Stock Trading Bot.

Usage:
	train.py <train-stock> <val-stock> [--strategy=<strategy>]
		[--window-size=<window-size>] [--batch-size=<batch-size>]
		[--episode-count=<episode-count>] [--model-name=<model-name>]
		[--pretrained] [--debug]

Options:
	--strategy=<strategy>             Q-learning strategy to use for training the network. Options:
																			`dqn` i.e. Vanilla DQN,
																			`t-dqn` i.e. DQN with fixed target distribution,
																			`double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
	--window-size=<window-size>       Size of the n-day window stock data representation
																		used as the feature vector. [default: 10]
	--batch-size=<batch-size>         Number of samples to train on in one mini-batch
																		during training. [default: 32]
	--episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
	--model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
	--pretrained                      Specifies whether to continue training a previously
																		trained model (reads `model-name`).
	--debug                           Specifies whether to use verbose logs during eval operation.
"""

"""
Model Training 
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""

import logging
import coloredlogs
import matplotlib.pyplot as plt
from docopt import docopt
import matplotlib.gridspec as gridspec
from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
		get_stock_data,
		format_currency,
		format_position,
		show_train_result,
		switch_k_backend_device
)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
				 strategy="t-dqn", model_name="model_debug", pretrained=False,
				 debug=False):
		""" Trains the stock trading bot using Deep Q-Learning.
		Please see https://arxiv.org/abs/1312.5602 for more details.

		Args: [python train.py --help]
		"""
		
		agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
		
		
		train_data = get_stock_data(train_stock)
		val_data = get_stock_data(val_stock)
		final_rewards = []
		train_roi = []
		valid_roi = []
		train_loss = []
		rewards = []
		initial_offset = val_data[1] - val_data[0]

		
		for episode in range(1, ep_count + 1):
				train_result,rewards = train_model(agent, episode, train_data, ep_count=ep_count,
																	 batch_size=batch_size, window_size=window_size)
				final_rewards.extend(rewards)
				train_roi.append(train_result[2])
				train_loss.append(train_result[3])
				val_result, _ = evaluate_model(agent, val_data, window_size, debug)
				valid_roi.append(val_result)
				show_train_result(train_result, val_result, initial_offset)

		gs = gridspec.GridSpec(2, 2)
		fig = plt.figure(figsize =(20,9))
		
		# To be shifted to Axis 1
		ax1 = fig.add_subplot(gs[0, 0])
		ax1.plot(range(len(train_loss)), train_loss, color='purple', label='loss')
		ax1.legend(loc=0, ncol=2, prop={'size':10}, fancybox=True, borderaxespad=0.)
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Train Loss')
		ax1.set_title('Loss w.r.t. Epochs')

		# To be shifted to Axis 2
		ax2 = fig.add_subplot(gs[0, 1])
		ax2.plot(range(len(train_roi)),train_roi, color = 'crimson', label='train')
		ax2.plot(range(len(valid_roi)), valid_roi, color='olive', label='val')
		ax2.legend(loc=0, ncol=2, prop={'size':10}, fancybox=True, borderaxespad=0.)
		ax2.set_ylabel('Return of Investment($)')
		ax2.set_xlabel('Epochs')
		ax2.set_title('Train and Valid ROI w.r.t. Epochs')

		# To be shifted to Axis 3
		ax3 = fig.add_subplot(gs[1, :])
		ax3.plot(range(len(final_rewards)),final_rewards, color='red', label = 'Reward of Rainbow DQN')
		ax3.set_xlabel('Episodes')
		ax3.set_ylabel('Rewards')
		ax3.set_title('Reward w.r.t. episodes')
		ax3.legend(loc=0, ncol=2, prop={'size':10}, fancybox=True, borderaxespad=0.)

		plt.show()

if __name__ == "__main__":

		args = docopt(__doc__)

		train_stock = args["<train-stock>"]
		val_stock = args["<val-stock>"]
		strategy = args["--strategy"]
		window_size = 10
		batch_size = 32
		ep_count = 30
		model_name = args["--model-name"]
		pretrained = args["--pretrained"]
		debug = args["--debug"]
	
		coloredlogs.install(level="DEBUG")

		switch_k_backend_device()

		try:
				main(train_stock, val_stock, window_size, batch_size,
						 ep_count, strategy=strategy, model_name=model_name, 
						 pretrained=pretrained, debug=debug)
		except KeyboardInterrupt:
				print("Aborted!")
