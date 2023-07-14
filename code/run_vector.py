import gymnasium as gym
from agents.Q_Agent import QLearningAgent
from gymnasium.vector import AsyncVectorEnv
from agents.Q_Agent_Vector import QLearningAgentVector


n_episodes = 50000


learning_rate=0.5              # 0 means the agent doesn't update q-table at all. 1 means the agent rewrites entire q-table. 
learning_decay_rate=0.0001
min_learning_rate=0.01
discount_factor=0.99            # gamma. 0 means agent only cares about immediate rewards. 1 means agent values future rewards equally to immediate rewards.
exploration_rate=1.0            # initial epsilon value
exploration_decay_rate=0.0001   # epsilon decay rate
min_exploration_rate=0.02
file_save_path="./"
num_bins=[30,30,40,20]                # Adjust the number of bins based on your environment. length depends on observation space variables, 
                                # the number size correlates to how granular you want the analog measurements to be digitized into.


if __name__ == "__main__":


    ##### VECTORIZING THE ENVIRONMENT #####


    # Define the environment name and number of parallel environments
    env_name = "CartPole-v1"
    num_envs = 4

    # Create the vectorized environment
    env = AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

    # Create an instance of the QLearningAgentVector
    agent = QLearningAgentVector(
        env, 
        num_envs,
        learning_rate,
        learning_decay_rate,
        min_learning_rate,
        discount_factor,
        exploration_rate,
        exploration_decay_rate,
        min_exploration_rate,
        file_save_path,
        num_bins
        )

    # Train the agent
    num_episodes = 10000
    num_steps = 200

    agent.train(num_episodes, num_steps)

    # Test the agent
    test_episodes = 10
    test_steps = 100

    test_rewards = agent.test(test_episodes, test_steps)
    avg_reward = sum(test_rewards) / len(test_rewards)

    print("Average test reward:", avg_reward)