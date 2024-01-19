import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
import random
from collections import deque
from tqdm.auto import tqdm



class DQNAgent:


	def __init__(
		self,
		env,
		learning_rate=0.8,             # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
		learning_decay_rate=0.00001,
		min_learning_rate=0.01,
		discount_factor=0.99,           # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
		exploration_rate=1.0,           # initial epsilon value
		exploration_decay_rate=0.0001,    # epsilon decay rate
		min_exploration_rate=0.02,
		file_save_path="./",
		input_model_path=None
	):
		
		self.env = env
		self.learning_rate = learning_rate
		self.learning_decay_rate = learning_decay_rate
		self.min_learning_rate = min_learning_rate
		self.discount_factor = discount_factor
		self.exploration_rate = exploration_rate
		self.exploration_decay_rate = exploration_decay_rate
		self.min_exploration_rate = min_exploration_rate

		self.file_save_path = file_save_path


		if input_model_path:
			self.load_model(input_model_path)
		else:
			# initialize q-table with zeros 
			self.model = self.create_model()



	def create_model(self):
	
		init = keras.initializers.HeUniform()
		model = keras.Sequential()
		model.add(keras.layers.Dense(24, input_shape=self.env.observation_space.shape, activation='relu', kernel_initializer=init))
		model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
		model.add(keras.layers.Dense(self.env.action_space.n, activation='linear', kernel_initializer=init))
		model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
		return model


	def load_model(self, model_path):
		with open(model_path, "rb") as handle:
			b = pickle.load(handle)
		self.model = b

		print("model loaded")


	def save_model(self, model):
		pass


	def train_part(self, replay_memory, target_model, done):
		MIN_REPLAY_SIZE = 1000
		if len(replay_memory) < MIN_REPLAY_SIZE:
			return
		
		batch_size = 64 * 2

		mini_batch = random.sample(replay_memory, batch_size)
		current_states = np.array([transition[0] for transition in mini_batch])
		current_qs_list = self.model.predict(current_states)
		new_current_states = np.array([transition[3] for transition in mini_batch])
		future_qs_list = target_model.predict(new_current_states)

		X = []
		Y = []

		for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
			if not done:
				max_future_q = reward + self.discount_factor * np.max(future_qs_list[index])
			else:
				max_future_q = reward
		
			current_qs = current_qs_list[index]
			current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q


			X.append(observation)
			Y.append(current_qs)

		self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


	def train(self, episodes):

		target_model = self.create_model()

		target_model.set_weights(self.model.get_weights())

		replay_memory = deque(maxlen=50_000)
		
		target_update_counter = 0


		X = []
		y = []

		steps_to_update_target_model = 0


		for i, epsiode in enumerate(pbar:= tqdm(range(episodes))):
			total_training_rewards = 0
			observation = self.env.reset()
			done = False
			while not done:
				steps_to_update_target_model += 1
				
				random_number = np.random.rand()
				if random_number <= self.exploration_rate:
					# explore
					action = self.env.action_space.sample()
				else:
					# perform best known action
					encoded = observation
					encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
					predicted = self.model.predict(encoded_reshaped).flatten()
					action = np.argmax(predicted)
				new_observation, reward, terminated, truncated, info = self.env.step(action)
				replay_memory.append([observation, action, reward,new_observation, done])

				if terminated or truncated:
					done = True

				# update main model every 4 steps
				if steps_to_update_target_model % 4 == 0 or done:
					self.train_part(replay_memory, target_model, done)

				pbar.set_postfix_str(f"Reward: {int(reward)}, Exploration: {self.exploration_rate}, Learning: {self.learning_rate}")

				observation = new_observation
				total_training_rewards += reward


				if done:
					print(f'Total training rewards: {total_training_rewards}, after {epsiode} episodes, with a final reward of {reward}')

					if steps_to_update_target_model >= 100:
						# print("copying main network weights to target")
						target_model.set_weights(self.model.get_weights())
						steps_to_update_target_model = 0
					break

			self.exploration_rate = self.min_exploration_rate + (1 - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * epsiode)
		self.env.close()


	def test(self, episodes, render=True):
		for epsiode in range(episodes):
			total_training_rewards = 0
			observation = self.env.reset()
			done = False
			while not done:
				if render:
					self.env.render()
				

				# perform best known action
				encoded = observation
				encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
				predicted = self.model.predict(encoded_reshaped).flatten()
				action = np.argmax(predicted)

				if done:
					print(f'Total training rewards: {total_training_rewards}, after {epsiode} episodes, with a final reward of {reward}')

				new_observation, reward, done, info = self.nv.step(action)