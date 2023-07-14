import bisect
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import h5py

class QLearningAgent:

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
        num_bins=[20,20],       # Adjust the number of bins based on your environment. length depends on observation space variables, the number size correlates to how granular you want the analog measurements to be digitized into.
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
        self.num_bins = num_bins

        # Create the state discretization boundaries for each dimension of the observation space
        self.discretization_boundaries = [np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i], num_bins[i] + 1).astype(np.float32) for i in range(len(num_bins))]

        if input_model_path:
            self.load_model(input_model_path)
        else:
            # initialize q-table with zeros 
            self.q_table = np.zeros(tuple(num_bins) + (self.env.action_space.n,))


    # Discretize a continuous observation to obtain a discrete state
    def discretize_state(self, observation):
        state = []
        for i in range(len(self.num_bins)):
            state.append(bisect.bisect(self.discretization_boundaries[i], observation[i]))
        return tuple(state)
    

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        q_value = self.q_table[state + (action,)]
        max_q_value = np.max(self.q_table[next_state])
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (reward + self.discount_factor * max_q_value)
        self.q_table[state + (action,)] = new_q_value


    def load_model(self, model_path):
        # with h5py.File(model_path, 'r') as hf:
        #     data = hf['q-table'][:]
        #     self.q_table = data
        self.q_table = np.load(model_path)

    def save_model(self, episode_number=None, threshold_params=None, score_avg=None):
        current_time = datetime.datetime.now()
        file_name = self.file_save_path + "q_table_" + str(episode_number) + "_" + current_time.strftime("%Y%m%d_%H%M%S")

        # with h5py.File(file_name + ".h5", 'w') as hf:
        #     hf.create_dataset('q-table', data=self.q_table)
        np.save(file_name + ".npy", self.q_table)

        # write an accompanying txt file with the training parameters and info relating to the newly saved model.
        file = open(file_name + ".txt", "w")
        file.write("agent_type=" + self.__class__.__name__ + "\n")
        file.write("env_type=" + self.env.spec.id + "\n")
        file.write(f"{self.learning_rate=}" + "\n")
        file.write(f"{self.learning_decay_rate=}" + "\n")
        file.write(f"{self.min_learning_rate=}" + "\n")
        file.write(f"{self.discount_factor=}" + "\n")
        file.write(f"{self.exploration_rate=}" + "\n")
        file.write(f"{self.exploration_decay_rate=}" + "\n")
        file.write(f"{self.min_exploration_rate=}" + "\n")
        file.write(f"{self.num_bins=}" + "\n")
        file.write("threshold_params [episodes_avg, avg_needed]=" + str(threshold_params) + "\n")
        file.write(f"{score_avg=}" + "\n")
        file.close()





    def train(self, num_episodes, threshold_params=None, show_graphs=False): # threshold_params is [episodes, score_avrg]
        rewards = []
        score_avg = 0

        # train for each episide
        for i, episode in enumerate(pbar:= tqdm(range(num_episodes))):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0

            # play one episode
            while not done:
                descrete_state = self.discretize_state(state)
                action = self.choose_action(descrete_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                descrete_next_state = self.discretize_state(next_state)
                self.update_q_table(descrete_state, action, reward, descrete_next_state)
                episode_reward += reward
                done = terminated or truncated
                state = next_state

            rewards.append(episode_reward)
            # update exploration rate
            self.exploration_rate = round(self.exploration_rate * (1 - self.exploration_decay_rate), 6)
            self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

            # update learning rate
            self.learning_rate = round(self.learning_rate * (1 - self.learning_decay_rate), 6)
            self.learning_rate = max(self.learning_rate, self.min_learning_rate)


            if len(rewards) > threshold_params[0]:
                score_avg = np.mean(rewards[-threshold_params[0]:]).round(2)
                pbar.set_postfix_str(f"Average Reward: {int(score_avg)}, Exploration: {self.exploration_rate}, Learning: {self.learning_rate}")

                # stop training when threshold is reached. Threshold is an average rewards over last x episodes.
                if threshold_params is not None and len(rewards) > threshold_params[0]:
                    if score_avg >= threshold_params[1]:
                        print("Threshold reached. Training stopped.")
                        break
            else:
                pbar.set_postfix_str(f"Avg Reward: n/a, Exploration: {round(self.exploration_rate, 2)}, Learning: {round(self.learning_rate, 2)}")


        self.save_model(i+1, threshold_params, score_avg)

        if show_graphs:
            rolling_length = 500
            fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
            axs[0].set_title("Episode rewards")
            reward_moving_average = (
                np.convolve(
                    np.array(self.env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
                )
                / rolling_length
            )
            axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
            axs[1].set_title("Episode lengths")
            length_moving_average = (
                np.convolve(
                    np.array(self.env.length_queue).flatten(), np.ones(rolling_length), mode="same"
                )
                / rolling_length
            )
            axs[1].plot(range(len(length_moving_average)), length_moving_average)
            plt.tight_layout()
            plt.show()
        



    def test(self, num_episodes, render=False):
        rewards = []

        for i, episode in enumerate(range(num_episodes)):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                descrete_state = self.discretize_state(state)
                action = self.choose_action(descrete_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state

                if render:
                    cv2.imshow("Test", self.env.render())
                    cv2.waitKey(int(1000/30))

            rewards.append(total_reward)
        
        # print(self.q_table)
        return rewards

