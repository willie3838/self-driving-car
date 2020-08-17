import gym
import highway_env
import numpy as np
import random
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.backend import manual_variable_initialization
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import time
from collections import deque


lost_his = []
eps_his =[]
manual_variable_initialization(True)

class DQN:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.001

        self.model = self.create_model()



    def create_model(self):
        try:
            model = Sequential()

            state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]

            model.add(Dense(24, input_dim=state_shape, activation="relu"))
            model.add(Dense(24, activation="relu"))
            model.add(Dense(24, activation="relu"))
            model.add(Dense(self.env.action_space.n))
            model.load_weights('C:/Users/willi/Documents/self_driving_car/models/highway/weights.h5')
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
            return model
        except:
            print("No loaded model, please select the right file")

    def level_one(self):
        try:
            model = Sequential()

            state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]

            model.add(Dense(128, input_dim=state_shape, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(self.env.action_space.n))
            model.load_weights('C:/Users/willi/Documents/self_driving_car/models/intersection/levelOneImprovedWeights.h5')
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
            return model
        except:
            print("No loaded model, please select the right file")

    def level_two(self):

        try:
            model = Sequential()

            state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]

            model.add(Dense(128, input_dim=state_shape, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(self.env.action_space.n))
            model.load_weights('C:/Users/willi/Documents/self_driving_car/models/intersection/levelTwo.h5')
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
            return model
        except:
                print("No loaded model, please select the right file")

    def act(self, state):
        return np.argmax(self.model.predict(state)[0])


def main():
    env = gym.make("highway-v0")
    EPISODES = 1000
    agent = DQN(env=env)

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        step = 0

        print("Episode %s"%episode)
        while True:
            env.render()
            state = np.array(state).ravel()
            state = state[np.newaxis,]
            action = agent.act(state)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            step+=1
            state = new_state
            if done:
                print(step)
                break

        print(episode_reward)



if __name__ == "__main__":
    main()