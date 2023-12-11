import gymnasium as gym
from agents.Q_Agent import QLearningAgent
from gymnasium.vector import AsyncVectorEnv
from agents.Q_Agent_Vector import QLearningAgentVector

n_episodes = 60000




learning_rate=0.7              # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
learning_decay_rate=0.0001
min_learning_rate=0.02
discount_factor=0.99            # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
exploration_rate=1.0            # initial epsilon value
exploration_decay_rate=0.0001   # epsilon decay rate
min_exploration_rate=0.02
file_save_path="./code/models/"
num_bins=[30,30,40,20]              # Adjust the number of bins based on your environment. length depends on observation space variables, 
                                    # the number size correlates to how granular you want the analog measurements to be digitized into.



# Create the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
# Create an instance of the QLearningAgent
agent = QLearningAgent(
    env, 
    learning_rate,
    learning_decay_rate,
    min_learning_rate,
    discount_factor,
    exploration_rate,
    exploration_decay_rate,
    min_exploration_rate,
    file_save_path,
    num_bins,
    # input_model_path="./code/models/q_table_39325_20231208_164116.pickle"
    )

# load pre-trained model
# agent.load_model("./models/q_table_27873_20230713_172532.h5.npy")

# Train the agent
# agent.train(num_episodes=n_episodes, threshold_params=[100, 400], show_graphs=True)


# load pre-trained model
agent.load_model("./code/models/q_table_39325_20231208_164116.pickle")


# Test the agent
rewards = agent.test(num_episodes=10, render=True)
average_reward = sum(rewards) / len(rewards)
print("Average reward:", average_reward)
