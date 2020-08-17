import gym
import highway_env
import numpy as np
import random
import keras
import os
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.backend import manual_variable_initialization
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque


lost_his = []
eps_his =[]
manual_variable_initialization(True)

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=15000)
        self.history = None
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.tau = 0.1
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.targetCounter = 0

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim = state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model



    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        UPDATE_TARGET_EVERY = 512
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        if self.targetCounter > UPDATE_TARGET_EVERY:
            self.target_train()
            self.targetCounter = 0

        samples = random.sample(self.memory, batch_size)

        state = np.zeros((batch_size, self.env.observation_space.shape[0] * self.env.observation_space.shape[1]))
        new_state = np.zeros((batch_size, self.env.observation_space.shape[0]* self.env.observation_space.shape[1]))
        action,reward,done = [],[],[]

        for i in range(batch_size):
            state[i] = samples[i][0]
            action.append(samples[i][1])
            reward.append(samples[i][2])
            new_state[i] = samples[i][3]
            done.append(samples[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(new_state)
        target_val = self.target_model.predict(new_state)

        for i in range(len(samples)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        self.history = self.model.fit(state, target, batch_size=batch_size, verbose=0)


        '''for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            target_next = self.model.predict(new_state)
            target_val = self.target_model.predict(new_state)
            if done:
                target[0][action] = reward
            else:
                a = np.argmax(target_next)
                target[0][action] = reward + self.gamma * (target_val[0][a])
            self.history = self.model.fit(state, target, epochs=1, verbose=0)'''


        self.targetCounter+=1


    def target_train(self):
        model_theta = self.model.get_weights()
        target_theta = self.target_model.get_weights()
        counter = 0
        for model_weights, target_weights in zip(model_theta, target_theta):
            target_weights = target_weights * (1-self.tau) + model_weights * self.tau
            target_theta[counter] = target_weights
            counter +=1
        self.target_model.set_weights(target_theta)


def plotReward():
    plt.plot(eps_his)
    plt.title('Model Rewards')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.show()

def plotLoss():
    plt.plot(lost_his)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('episodes')
    plt.show()

def main():
    env = gym.make("intersection-v0")
    EPISODES = 1000000
    agent = DQN(env=env)
    reward_counter = 0
    print(env.action_space.n)
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        step = 0
        while True:
            state = np.array(state).ravel()
            state = state[np.newaxis,]

            action = agent.act(state)
            action2 = 0
            new_state, reward, done, _ = env.step(action, action2)
            episode_reward += reward

            new_state = np.array(new_state).ravel()
            new_state = new_state[np.newaxis,]
            agent.remember(state, action, reward, new_state, done)
            agent.replay()
            state = new_state
            step+=1
            if done:
                print(step)
                break
        if episode_reward >= 1   :
            reward_counter+=1
        else:
            reward_counter = 0
        print("Episode %s" % episode)
        print("Score: %s" % episode_reward)
        print("Reward Counter: %s" % reward_counter)
        if reward_counter >= 500:
            break

        if agent.history != None:
            lost_his.append(agent.history.history['loss'])

        eps_his.append(episode_reward)

    filename = 'D:/Documents/self_driving_car/models/intersection/levelOneModel.h5'
    save_model(agent.target_model, filename)
    agent.target_model.save_weights('D:/Documents/self_driving_car/models/intersection/levelOne.h5')
    print("Model Saved!")
    plotLoss()
    plotReward()





if __name__ == "__main__":
    main()