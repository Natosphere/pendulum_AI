import bisect
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

###### TODO: bring this vectorized version to parity with the regular version.  #######


class QLearningAgentVector:
    def __init__(
        self,
        env,
        num_envs,
        learning_rate=0.5,             # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
        learning_decay_rate=0.0001,
        min_learning_rate=0.01,
        discount_factor=0.99,           # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
        exploration_rate=1.0,           # initial epsilon value
        exploration_decay_rate=0.0001,    # epsilon decay rate
        min_exploration_rate=0.02,
        file_save_path="./",
        num_bins=[20,20]       # Adjust the number of bins based on your environment. length depends on observation space variables, the number size correlates to how granular you want the analog measurements to be digitized into.
    ):

        self.env = env
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        
        self.file_save_path = file_save_path
        self.num_bins = num_bins

        # Initialize the Q-table with zeros
        self.discretization_boundaries = [np.linspace(self.env.single_observation_space.low[i], self.env.single_observation_space.high[i], num_bins[i] + 1).astype(np.float32) for i in range(len(num_bins))]
        
        # initialize q-table with zeros 
        self.q_table = np.zeros(tuple(num_bins) + (self.env.single_action_space.n,))

        
    # Discretize a continuous observation to obtain a discrete state
    def discretize_state(self, observation):
        state = []
        for i in range(len(self.num_bins)):
            state.append(bisect.bisect(self.discretization_boundaries[i], observation[i]))
        return tuple(state)

    
    def choose_action(self, state):
        # Exploration vs. exploitation trade-off
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Explore - choose a random action
            action = self.env.action_space.sample()
        else:
            # Exploit - choose the action with maximum Q-value for each state
            action = np.argmax(self.q_table[state], axis=1)
        return action
    
    def update_q_table(self, states, actions, rewards, next_states, terminated, truncated):
        # Update the Q-values of the (state, action) pairs using the Q-learning formula
        q_values = self.q_table[states, actions]
        max_q_values = np.max(self.q_table[next_states], axis=1)
        new_q_values = (1 - self.learning_rate) * q_values + self.learning_rate * (rewards + self.discount_factor * max_q_values * (1 - (terminated or truncated)))
        self.q_table[states, actions] = new_q_values
        
    def train(self, num_episodes, num_steps, threshold_params=None, show_graphs=False):
        rewards = []

        for i, episode in enumerate(pbar:= tqdm(range(num_episodes))):
            states = self.env.reset()[0]
            # states = [x[0] for x in states]
            episode_reward = 0

            for _ in range(num_steps):
                discretized_states = []
                actions = []
                discretized_next_states = []
                # loop through all envs
                for i in range(self.num_envs):
                    discretized_states.append(self.discretize_state(states[i]))
                    actions.append(self.choose_action([discretized_states[i]]))

                next_states, rewards, terminated, truncated, _ = self.env.step(actions)

                # loop through all envs
                for i in enumerate(range(self.num_envs)):
                    discretized_next_states.append(self.discretize_state(next_states[i]))

                self.update_q_table(discretized_states, actions, rewards, discretized_next_states, terminated, truncated)
                episode_reward += [reward for reward in rewards]
                states = next_states


                
            rewards.append(episode_reward)
            # Decay exploration rate after each episode
            self.exploration_rate *= self.exploration_decay_rate
            self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
            
            # Decay learning rate after each episode
            self.learning_rate *= self.exploration_decay_rate
            self.learning_rate = max(self.learning_rate, self.min_learning_rate)


            if len(rewards) > threshold_params[0]:
                score_avg = np.mean(rewards[-threshold_params[0] * self.num_envs:]).round(2)
                pbar.set_postfix_str(f"Average Reward: {int(score_avg)}, Exploration: {self.exploration_rate}, Learning: {self.learning_rate}")

                # stop training when threshold is reached. Threshold is an average rewards over last x episodes.
                if threshold_params is not None and len(rewards) > threshold_params[0]:
                    if score_avg >= threshold_params[1]:
                        print("Threshold reached. Training stopped.")
                        break
            else:
                pbar.set_postfix_str(f"Avg Reward: n/a, Exploration: {round(self.exploration_rate, 2)}, Learning: {round(self.learning_rate, 2)}")


            
    def test(self, num_episodes, num_steps):
        rewards = []
        
        for episode in range(num_episodes):
            states = self.env.reset()
            total_reward = 0
            
            for _ in range(num_steps):
                actions = np.argmax(self.q_table[states], axis=1)
                next_states, episode_rewards, dones, _ = self.env.step(actions)
                total_reward += np.sum(episode_rewards)
                states = next_states
                
            rewards.append(total_reward)
            
        return rewards