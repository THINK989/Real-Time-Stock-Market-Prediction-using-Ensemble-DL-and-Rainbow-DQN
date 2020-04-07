"""
Agent
The main Trading Agent implementation.
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""
import random
from collections import deque
import heapq
import numpy as np
import tensorflow as tf
import keras.backend as K
from itertools import count
from keras.models import Sequential, Model
from keras.models import load_model, clone_model
from keras.layers import Dense, Lambda, Input, Add
from keras.optimizers import Adam
from .NoisyDense import NoisyDense
from keras.engine.topology import Layer
from keras.utils import CustomObjectScope


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None, manual = False):
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.buffer = []
        self.first_iter = True
        self.nstep = 5
        self.n_step_buffer = deque(maxlen = self.nstep)
        self.cnt = count()
        self.alpha = 0.6
        # self.memory = deque(maxlen = 10000)
        
        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # For Categorical DQN
        #Initializing Atoms
        # self.num_atoms = 51
        # self.v_max = 30 # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        # self.v_min = -10 # -0.1*26 - 1 = -3.6
        # self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        # self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load(manual)
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            with CustomObjectScope({"NoisyDense":NoisyDense}):
            # target network
                self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    # def _model(self):
    #     """Creates the model
    #     """
    #     model = Sequential()
    #     model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
    #     model.add(Dense(units=256, activation="relu"))
    #     model.add(Dense(units=256, activation="relu"))
    #     model.add(Dense(units=128, activation="relu"))
    #     model.add(Dense(units=self.action_size))

    #     model.compile(loss=self.loss, optimizer=self.optimizer)
    #     return model

    #Dueling Model with Noisy Layers
    def _model(self):


        X_input = Input((self.state_size,))
        X = Dense(units = 128, activation="relu",input_dim=self.state_size)(X_input)
        X = Dense(units = 256, activation="relu")(X)
        X = Dense(units = 256, activation="relu")(X)
        X = NoisyDense(units = 128, activation="relu")(X)
        state_value = NoisyDense(1, activation = "linear")(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

        action_advantage = NoisyDense(self.action_size, activation = "linear")(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.action_size,))(action_advantage)

        out = Add()([state_value, action_advantage])

        model = Model(inputs = X_input, outputs = out)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        model.summary()
        return model



    def remember(self, state, action, reward, next_state, done, td_error):
        """Adds relevant data to memory
        """
        # self.memory.append((state, action, reward, next_state, done))

        # n-step queue for calculating return of n previous steps
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.nstep:
          return
        
        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        
        l_state, l_action = self.n_step_buffer[0][:2]

        t = (l_state, l_action, l_reward, l_next_state, l_done)
        heapq.heappush(self.buffer, (-td_error, next(self.cnt), t))
        if len(self.buffer) > 100000:
            self.buffer = self.buffer[:-1]
        
        heapq.heapify(self.buffer)

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        # for categorical DQN 
        # z = self.model.predict(state) # Return a list [1x51, 1x51, 1x51]

        # z_concat = np.vstack(z)
        # q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 

        # # Pick action with the biggest Q value
        # action_idx = np.argmax(q)
        
        # return action_idx

        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
         # Semi Stochastic Prioritization
        prioritization = int(batch_size*self.alpha)
        batch_prioritized = heapq.nsmallest(prioritization, self.buffer)
        batch_uniform = random.sample(self.buffer, batch_size-prioritization)
        batch = batch_prioritized + batch_uniform
        # print(self.buffer)
        batch = [e for (_, _, e) in batch]
        #print(batch)


        # mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])#1
                y_train.append(q_values[0])#2
                
        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]#3

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def calculate_td_error(self, state, action, reward, next_state, done):
        if not done:
            target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]
        else:
            target = reward

        q_values = self.model.predict(state)[0][action]
        
        return q_values - target

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self, manual):
        if manual:
            return load_model( 'models/' + self.model_name,custom_objects={'NoisyDense': NoisyDense})
        else:
            return load_model( 'rainbow/models/' + self.model_name,custom_objects={'NoisyDense': NoisyDense})