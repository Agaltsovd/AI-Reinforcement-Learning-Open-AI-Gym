import tflearn
from collections import deque
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from actor_network import Network_Actor
from critic_network import Network_Critic


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
class BufferReplay(object):
    
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, t, next_state):
        experience = (state, action, reward, t, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, sizeBatch):
        batch = []

        if self.count < sizeBatch:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, sizeBatch)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, t_batch, next_state_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        


def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):
    
    sess.run(tf.global_variables_initializer())

    # Initializiaton weights of the target network
    actor.update_targetNetwork()
    critic.update_targetNetwork()

    # Initializiation of the replay memory or buffer replay
    buffer_replay = BufferReplay(buffer_size, 0)

    max_episodes = ep
    max_steps = 1000
    score_list = []

    for i in range(max_episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            env.render()

            action = actor.predict(np.reshape(state, (1, actor.state_space))) + actor_noise()
            next_state, reward, done, info = env.step(action[0])
            buffer_replay.add(np.reshape(state, (actor.state_space,)), np.reshape(action, (actor.action_space,)), reward,
                              done, np.reshape(next_state, (actor.state_space,)))

            # update network using batches
            if buffer_replay.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = buffer_replay.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # updating critic network
            predicted_q_value, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))

            # Updating actor network policy using gradients
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Updating target networks of critic and actor networks
            actor.update_targetNetwork()
            critic.update_targetNetwork()

            state = next_state
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(int(score), i, max_episodes))
                break

        score_list.append(score)

    return score_list


if __name__ == '__main__':

    with tf.Session() as sess:

        env = gym.make('Pendulum-v0')

        env.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        ep = 400
        tau_factor = 0.001
        gamma = 0.99
        minimum_batch = 64
        actor_learningRate = 0.0001
        critic_learningRate = 0.001
        buffer_size = 1000000
       

        state_space = env.observation_space.shape[0]
        action_space = env.action_space.shape[0]
        action_threshold = env.action_space.high

        noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_space))
        networkActor = Network_Actor(sess, state_space, action_space, action_threshold, actor_learningRate, tau_factor, minimum_batch)
        networkCritic = Network_Critic(sess, state_space, action_space, critic_learningRate, tau_factor, gamma, networkActor.get_numberOfVariables())

        scores = train(sess, env, networkActor, networkCritic, noise, buffer_size, minimum_batch, ep)
        plt.plot([i + 1 for i in range(0, ep, 3)], scores[::3])
        plt.show()
        
    