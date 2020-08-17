import gym
import highway_env
import numpy as np
import pandas as pd
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
        self.learning_rate = 0.01
        self.tau = 0.1
        '''self.model = self.create_model()
        self.target_model = self.create_model()'''
        self.levelOne = self.level_one()
        self.levelTwo = self.level_two()
        self.levelPredict = self.level_predict()
        self.levelPredict_target = self.level_predict()
        self.targetCounter = 0

        self.critical = self.critical()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim = state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def level_two(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwoOldObservationWeights.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def level_one(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelOneOldObservationWeights.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def level_predict(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * (self.env.observation_space.shape[1]+1)
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2,activation="softmax"))
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/data/levelWeights.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def critical(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/data/criticalWeights.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model


def plotReward():
    plt.figure(0)
    plt.plot(eps_his)
    plt.title('Model Rewards')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.savefig('/Users/williamchan/Documents/side_projects/self_driving_car/graphs/DQN/intersection/creep/levelPredictRewardTwoStatic.png')

def plotLoss():
    plt.figure(1)
    plt.plot(lost_his)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('episodes')
    plt.savefig('/Users/williamchan/Documents/side_projects/self_driving_car/graphs/DQN/intersection/creep/levelPredictLossTwoStatic.png')



def main():
    env = gym.make("intersection-v0")
    EPISODES = 100000000
    BETA = 0.6
    BASE_PROB = 0.5
    prob_two = BASE_PROB
    agent = DQN(env=env)
    reward_counter = 0



    print(env.action_space.n)
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        step = 0
        while True:
            env.render()
            state = np.array(state).ravel()
            state = state[np.newaxis,]

            action2 = np.argmax(agent.levelOne.predict(state)[0])
            action3 = np.argmax(agent.levelOne.predict(state)[0])

            ######### CREATING NEW OBSERVATION ##################
            prevState = state
            prevState = prevState.reshape(3, 4)
            action1_ = np.array([0])
            action2_ = np.array([action2])
            action3_ = np.array([action3])
            actions = np.vstack((action2_, action3_, action1_))
            prevState = np.concatenate((prevState, actions), axis=1)
            prevState = prevState.ravel()
            prevState = prevState[np.newaxis]
            ############ CREATING NEW OBSERVATION #################

            if(np.argmax(agent.critical.predict(state)) == 1):
                print("Critical")
                if(np.argmax(agent.levelPredict.predict(prevState))==1):
                    prob_two = (1-BETA)*prob_two + BETA*1
                else:
                    prob_two = (1 - BETA) * prob_two + BETA * 0
                if prob_two >= 0.5:
                    print("Level Two")
                    action = np.argmax(agent.levelOne.predict(state)[0])
                else:
                    print("Level One")
                    action = np.argmax(agent.levelTwo.predict(state)[0])
            else:
                print("Not critical")
                action = np.argmax(agent.levelOne.predict(state)[0])


            print(action)


            new_state, reward, done, _ = env.step(action, action2, action3)
            episode_reward += reward

            state = new_state
            step+=1
            if done:
                print("Steps %s"%(step-1))
                break
        '''if episode_reward >= 0.92:
            reward_counter+=1
        else:
            reward_counter = 0'''
        print("Episode %s" % episode)
        print("Reward: %s" % episode_reward)
        '''print("Reward Counter: %s" % reward_counter)
        if reward_counter >= 100:
            break'''


        eps_his.append(episode_reward)

    filename = "/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelPredictModelTwoStatic.h5.h5"
    save_model(agent.levelPredict, filename)
    agent.levelPredict.save_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelPredictTwoStatic.h5')
    print("Model Saved!")

    plotReward()





if __name__ == "__main__":
    main()