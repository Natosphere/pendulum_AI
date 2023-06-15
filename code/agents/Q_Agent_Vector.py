import numpy as np


###### TODO: bring this vectorized version to parity with the regular version.  #######


class QLearningAgentVector:
    def __init__(
        self, 
        env, 
        num_envs, 
        learning_rate=0.01,             # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
        min_learning_rate=0.005,
        discount_factor=0.99,           # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
        exploration_rate=1.0,           # initial epsilon value
        exploration_decay_rate=0.01,    # epsilon decay rate
        min_exploration_rate=0.01, 
    ):

        self.env = env
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        self.min_learning_rate = min_learning_rate
        
        # Initialize the Q-table with zeros
        self.q_table = np.zeros((self.env.num_envs, self.env.action_space.n))
        
    def choose_action(self, states):
        # Exploration vs. exploitation trade-off
        if np.random.uniform(0, 1, size=self.num_envs) < self.exploration_rate:
            # Explore - choose a random action
            actions = self.env.action_space.sample(self.num_envs)
        else:
            # Exploit - choose the action with maximum Q-value for each state
            actions = np.argmax(self.q_table[states], axis=1)
        return actions
    
    def update_q_table(self, states, actions, rewards, next_states, terminated, truncated):
        # Update the Q-values of the (state, action) pairs using the Q-learning formula
        q_values = self.q_table[states, actions]
        max_q_values = np.max(self.q_table[next_states], axis=1)
        new_q_values = (1 - self.learning_rate) * q_values + self.learning_rate * (rewards + self.discount_factor * max_q_values * (1 - (terminated or truncated)))
        self.q_table[states, actions] = new_q_values
        
    def train(self, num_episodes, num_steps):
        for episode in range(num_episodes):
            states = self.env.reset()
            for _ in range(num_steps):
                actions = self.choose_action(states)
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                self.update_q_table(states, actions, rewards, next_states, terminated, truncated)
                states = next_states
                
            # Decay exploration rate after each episode
            self.exploration_rate *= self.exploration_decay_rate
            self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
            
            # Decay learning rate after each episode
            self.learning_rate *= self.exploration_decay_rate
            self.learning_rate = max(self.learning_rate, self.min_learning_rate)
            
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