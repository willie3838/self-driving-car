import gym
import highway_env
import numpy as np
import pandas as pd
import random
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.backend import manual_variable_initialization



lost_his = []
eps_his =[]
manual_variable_initialization(True)

class DQN:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.001

    def create_model(self):
        try:
            model = Sequential()

            state_shape = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]

            model.add(Dense(128, input_dim=state_shape, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(self.env.action_space.n))
            model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwo.h5')
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
            return model
        except:

            print("No loaded model, please select the right file")
    def level_predict(self):
        try:
            model = Sequential()
            state_shape = self.env.observation_space.shape[0] * (self.env.observation_space.shape[1]+1)
            model.add(Dense(128, input_dim=state_shape, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(2, activation="softmax"))
            model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelPredictTwoStatic.h5')
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
            model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelOneOldObservationWeights.h5')
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
            model.load_weights('/Users/williamchan/Documents/side_projects/self_driving_car/models/intersection/creep/levelTwoOldObservationWeights.h5')
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
            return model
        except:
                print("No loaded model, please select the right file")


def main():
    env = gym.make("intersection-v0")
    EPISODES = 1
    agent = DQN(env=env)
    data = pd.read_pickle('data.pkl')
    #data = pd.DataFrame({'Observations':[],'Observations2':[],'Critical':[],"Action1":[],"Action2":[],"Level":[]})

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        step = 0
        vehicle = random.randint(1, 2)

        print("Episode %s"%episode)
        while True:
            #env.render()
            state = np.array(state).ravel()
            state = state[np.newaxis,]

            action = np.argmax(agent.level_one().predict(state)[0])
            action2 = np.argmax(agent.level_one().predict(state)[0])
            action3 = np.argmax(agent.level_one().predict(state)[0])

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

            test1 = np.argmax(agent.level_one().predict(state)[0])
            test2 = np.argmax(agent.level_two().predict(state)[0])
            print(test1)
            print(test2)

            if(test1 == test2):
                data = data.append({'Observations':state.tolist(),"Observations2":prevState.tolist(),"Critical":0,"Action1":action2, "Action2":action3,"Level":1}, ignore_index=True)
            else:
                data = data.append({'Observations':state.tolist(),"Observations2":prevState.tolist(),"Critical":1,"Action1":action2, "Action2":action3,"Level":1},ignore_index=True)

            new_state, reward, done, _ = env.step(action, action2, action3)


            episode_reward += reward
            step+=1
            state = new_state
            if done:
                print(step)
                break
        print(data)
        data.to_pickle("data.pkl")

        print(episode_reward)



if __name__ == "__main__":
    main()