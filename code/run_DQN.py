import gymnasium as gym
from agents.DQN_Agent import DQNAgent



n_episodes = 300



learning_rate=0.7              # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
learning_decay_rate=0.0001
min_learning_rate=0.02
discount_factor=0.99            # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
exploration_rate=1.0            # initial epsilon value
exploration_decay_rate=0.0001   # epsilon decay rate
min_exploration_rate=0.02
file_save_path="./code/models/"




# Create the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
# Create an instance of the DQN_agent
agent = DQNAgent(
	env, 
	learning_rate,
	learning_decay_rate,
	min_learning_rate,
	discount_factor,
	exploration_rate,
	exploration_decay_rate,
	min_exploration_rate,
	file_save_path,
	# input_model_path="./code/models/q_table_39325_20231208_164116.pickle"
)



# Train the agent
agent.train(episodes=n_episodes)


# load pre-trained model
# agent.load_model("./code/models/q_table_39325_20231208_164116.pickle")


# Test the agent
rewards = agent.test(episodes=10, render=True)
average_reward = sum(rewards) / len(rewards)
print("Average reward:", average_reward)