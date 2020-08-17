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
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.tau = 0.1
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.levelOne = self.level_one()
        #self.levelTwo = self.level_two()
        self.targetCounter = 0

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
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwo.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def level_one(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        model.add(Dense(128, input_dim=state_shape, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelOne.h5')
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()#return random.randint(0,1)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action,reward, new_state, done])

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
        action,reward,done,levelPredict = [],[],[],[]

        for i in range(batch_size):
            state[i] = samples[i][0]
            action.append(samples[i][1])
            #levelPredict.append(samples[i][2])
            reward.append(samples[i][2])
            new_state[i] = samples[i][3]
            done.append(samples[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(new_state)
        target_val = self.target_model.predict(new_state)
        '''target = self.levelPredict.predict(state)
        target_next = self.levelPredict.predict(new_state)
        target_val = self.levelPredict_target.predict(new_state)'''

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
        '''model_theta = self.model.get_weights()
        target_theta = self.target_model.get_weights()
        counter = 0
        for model_weights, target_weights in zip(model_theta, target_theta):
            target_weights = target_weights * (1-self.tau) + model_weights * self.tau
            target_theta[counter] = target_weights
            counter +=1
        self.target_model.set_weights(target_theta)'''

        model_theta = self.model.get_weights()
        target_theta = self.target_model.get_weights()
        counter = 0
        for model_weights, target_weights in zip(model_theta, target_theta):
            target_weights = target_weights * (1 - self.tau) + model_weights * self.tau
            target_theta[counter] = target_weights
            counter += 1
        self.target_model.set_weights(target_theta)


def plotReward():
    plt.figure(0)
    plt.plot(eps_his)
    plt.title('Model Rewards')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.savefig('/Users/williamchan/Documents/side_projects/self_driving_car/graphs/DQN/intersection/creep/levelTwoRewardOneVehicle.png')

def plotLoss():
    plt.figure(1)
    plt.plot(lost_his)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('episodes')
    plt.savefig('/Users/williamchan/Documents/side_projects/self_driving_car/graphs/DQN/intersection/creep/levelTwoLossOneVehicle.png')

def evaluate(agent, env):
    if(agent.epsilon == agent.epsilon_min):
        eval_reward = []
        for episode in range(5):
            state = env.reset()
            vehicle1_state = env.resetOne()
            episode_reward = 0

            step = 0
            while True:
                # env.render()
                state = np.array(state).ravel()
                state = state[np.newaxis,]
                vehicle1_state = np.array(vehicle1_state).ravel()
                vehicle1_state = vehicle1_state[np.newaxis,]


                action2 = np.argmax(agent.levelOne.predict(vehicle1_state)[0])
                action3 = np.argmax(agent.levelOne.predict(vehicle1_state)[0])
                action = np.argmax(agent.model.predict(state)[0])

                new_state, reward, done, new_vehicle1_state, _ = env.step(action, action2, action3)
                episode_reward += reward

                new_state = np.array(new_state).ravel()
                new_state = new_state[np.newaxis,]

                vehicle1_state = new_vehicle1_state

                state = new_state
                step += 1
                if done:
                    break
            eval_reward.append(episode_reward)
        print(eval_reward)
        print(np.mean(eval_reward))
        return np.mean(eval_reward)
    return 0



def main():
    env = gym.make("intersection-v0")
    EPISODES = 100000000
    agent = DQN(env=env)
    reward_counter = 0


    print(env.action_space.n)
    for episode in range(EPISODES):
        state = env.reset()
        vehicle1_state = env.resetOne()
        episode_reward = 0

        step = 0
        while True:
            env.render()
            state = np.array(state).ravel()
            state = state[np.newaxis,]

            vehicle1_state = np.array(vehicle1_state).ravel()
            vehicle1_state = vehicle1_state[np.newaxis,]
            print(vehicle1_state)

            action2 = np.argmax(agent.levelOne.predict(vehicle1_state)[0])
            action3 = np.argmax(agent.levelOne.predict(vehicle1_state)[0])
            action = agent.act(state)

            new_state, reward, done,new_vehicle1_state, _ = env.step(action, action2, action3)
            episode_reward += reward

            new_state = np.array(new_state).ravel()
            new_state = new_state[np.newaxis,]

            agent.remember(state, action, reward, new_state, done)
            agent.replay()

            state = new_state
            vehicle1_state = new_vehicle1_state
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
        if agent.history != None:
            lost_his.append(agent.history.history['loss'])
        if evaluate(agent, env) >= 0.95:
            break

        eps_his.append(episode_reward)

    filename = "/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwoModelOneVehicle.h5"
    save_model(agent.model, filename)
    agent.model.save_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwoWeightsOneVehicle.h5')
    print("Model Saved!")
    plotLoss()
    plotReward()





if __name__ == "__main__":
    main()