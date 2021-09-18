# CartPole-v1 OpenAI Submission
# Artificial Intelligence
# Team: Denny + Daniyar + Ilias
import gym
import numpy as np
import random
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from collections import deque
import matplotlib.pyplot as plt

def makeNNModel(input_shape, action_space):
    input_to_model = Input(input_shape) #the size is 4 (4 inputs)
    # 3 layers with nodes 512, 256 and 64
    dense_input_to_model = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(input_to_model)
    dense_input_to_model = Dense(256, activation="relu", kernel_initializer='he_uniform')(dense_input_to_model)
    dense_input_to_model = Dense(64, activation="relu", kernel_initializer='he_uniform')(dense_input_to_model)
    dense_input_to_model = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(dense_input_to_model)
    model = Model(inputs = input_to_model, outputs = dense_input_to_model, name='DQN-Model-CartPole')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"]) #hyperparameters tuned
    model.summary() #display the model's information
    return model

class CartPoleAgentDeepQN:
    def __init__(self):
        self.env = gym.make('CartPole-v1') # openai gym env cartpole-v1
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.no_of_episodes = 1000 #change from default 500
        self.memory = deque(maxlen=2000) #maximum number of observations the memory can hold for replying / re-training
        
        self.epsilon = 1.0  # exploration rate for explore-exploit
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.gamma = 0.95    # discount rate
        self.batch_size = 64 #batch size for 
        self.train_start = 1000 # for exploration decay decision
        # create main model
        self.model = makeNNModel(input_shape=(self.state_size,), action_space = self.action_size) # make the model

    def getSavedModel(self, h5FileName):
        self.model = load_model(h5FileName)

    def saveModel(self, h5FileName):
        self.model.save(h5FileName)

    #choose between exploration and exploitation
    def makeActionDecision(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state)) #exploitation
        else:
            return random.randrange(self.action_size) #exploration

    # sample old observations from memory and to re-train the model in replay()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        action, reward, done = [], [], []
        # Randomly sample minibatch from the memory
        sampledFromMemory = random.sample(self.memory, min(len(self.memory), self.batch_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        state = np.zeros((self.batch_size, self.state_size))

        for i in range(self.batch_size):
            state[i] = sampledFromMemory[i][0]
            action.append(sampledFromMemory[i][1])
            reward.append(sampledFromMemory[i][2])
            next_state[i] = sampledFromMemory[i][3]
            done.append(sampledFromMemory[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state) #prediction for the batch above

        #updating Q values
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i])) #choose max among all next actions, discount it and add to immediate reward
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0) # train NN
            
    def trainModel(self):
        loss = []
        eps = 0
        for e in range(self.no_of_episodes):
            eps +=1
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            score = 0
            while not done:
                self.env.render()
                action = self.makeActionDecision(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                score += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("For Episode: {} out of a maximum of {}, the score is: {}, epsilon is: {:.2}".format(e, self.no_of_episodes, i, self.epsilon))
                    if i == 500:
                        self.saveModel("savedDqnModelCartPole.h5")
                        print("The model has been trained and saved as an H5 file for testing to check the score it can obtain.")
                        return
                self.replay()
            loss.append(score)
        return loss, eps

    # to test the model, use this
    def testModel(self):
        self.getSavedModel("savedDqnModelCartPole.h5")
        for e in range(self.no_of_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("For Episode: {} out of a maximum of {}, the score is: {}".format(e, self.no_of_episodes, i))
                    break

if __name__ == "__main__":
    dqnAgent = CartPoleAgentDeepQN()
    loss, eps = dqnAgent.trainModel() # use loss and eps to plot the performance of the model while tuning hyperparameters

    #toggle between .run() and .test() after the .h5 is saved
    # dqnAgent.testModel()