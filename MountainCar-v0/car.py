import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Hyper Parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 0.8
MIN_EPSILON = 0.05
DECAY_OPTION = 2    #   1-optimal, 2-div by 1.5, 3- no reduction
EPISODES_NUM = 20000

steps_taken_list = []


def updateEpsilon(option, epsilon):
    if option == 1 and epsilon > MIN_EPSILON:
        # Calculate episodic reduction in epsilon
        epsilon -= (epsilon - MIN_EPSILON) / EPISODES_NUM
    elif option == 2:
        epsilon /=1.5
    return epsilon

def initQTable(env):
    # Make table based on state space
    print(env.observation_space.high)
    print(env.observation_space.low)
    state_num = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    state_num = np.round(state_num, 0).astype(int) + 1

    # Initialize Q table
    return np.random.uniform(low=-1, high=1, size=(state_num[0], state_num[1], env.action_space.n))


def QLearn(env, learning, discount, epsilon, episodes):

    # make a new Q table
    QTable = initQTable(env)

    # arrays for plotting rewards
    reward_list = []
    ave_reward_list = []
    max_reward_list = []
    min_reward_list = []

    # Main Q learning loop
    for i in range(episodes):
        # reset parameters
        steps_total = 0
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        adj_state = (state - env.observation_space.low) * np.array([10, 100])
        adj_state = np.round(adj_state, 0).astype(int)

        while done != True:
            # Show last 10 episodes
            if i >= (episodes - 10):
                env.render()

            if np.random.random() < 1 - epsilon:
                # exploit
                action = np.argmax(QTable[adj_state[0], adj_state[1]])
            else:
                # explore
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            steps_total += 1
            state, reward, done, info = env.step(action)

            # States
            state_adj = (state - env.observation_space.low) * np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            # Reached the flag
            if done and state[0] >= 0.5:
                QTable[adj_state[0], adj_state[1], action] = reward

            # Update Q value with Bellman equation
            else:
                dQ = learning * (reward +
                                    discount * np.max(QTable[state_adj[0],
                                                        state_adj[1]]) -
                                    QTable[adj_state[0], adj_state[1], action])
                QTable[adj_state[0], adj_state[1], action] += dQ

            # Update variables
            tot_reward += reward
            adj_state = state_adj

        steps_taken_list.append(steps_total)

        # Decay epsilon
        epsilon = updateEpsilon(DECAY_OPTION, epsilon)

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            max_reward_list.append(np.max(reward_list))
            ave_reward_list.append(ave_reward)
            min_reward_list.append(np.min(reward_list))
            reward_list = []

    env.close()

    return ave_reward_list, min_reward_list, max_reward_list


# Run Q-learning algorithm
ave_reward_list, min_reward_list, max_reward_list = QLearn(env, LEARNING_RATE, DISCOUNT, EPSILON, EPISODES_NUM)

print(steps_taken_list)

# Plot Rewards
plt.plot(100 * (np.arange(len(ave_reward_list)) + 1), ave_reward_list, label='Average')
plt.plot(100 * (np.arange(len(min_reward_list)) + 1), min_reward_list, label='Minimum')
plt.plot(100 * (np.arange(len(max_reward_list)) + 1), max_reward_list, label='Maximum')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward vs Episodes')
plt.savefig('Results.jpg')
plt.close()