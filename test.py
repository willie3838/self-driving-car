import gym
import numpy as np
import random
import tensorflow as tf
import highway_env

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 1000

class DeepQNetwork:
    def __init__(self, action_space, obs_space, learning_rate =0.01,
                 discount=0.8, e_greedy=0.9, target_update=300, memory_size=500,
                 batch_size=32, e_greedy_increment = None,output_graph=True):

        self.action_space = action_space
        self.obs_space = obs_space
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon_max = e_greedy
        self.target_update = target_update
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # used to track the amount of iterations until target network update
        self.target_update_counter = 0

        # why is it 500 x 10 you only need [s,a,r,s_] ?????????
        self.memory = np.zeros((self.memory_size, obs_space *2+2))

        # builds the target and evaluation neural net
        self._build_net()

        # WHAT DOES THIS DO???? I think it gets the weights/biases of the neural nets?
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # operation to update the target network through using zip which returns a tuple of the weights
        # thsi then assigns the evaluation net's weights to the target's weights
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t,e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()


        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []


    def _build_net(self):
        # inputs
        self.s = tf.placeholder(tf.float32, [None, self.obs_space], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.obs_space], name='s_')
        self.r = tf.placeholder(tf.float32, [None,], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 125, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.action_space, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 125, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.action_space, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            # tf.reduce_max takes the maximum element in axis 1, meaning the row. so basically it takes
            # the maximum q value
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            # tf.stop_gradient used to make tf to ignore the inputs of this tensor when computing the gradient
            self.q_target = tf.stop_gradient(q_target)

        ## FIGURE OUT WTF THE REST OF THE CODE DOES FROM HERE
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name = 'TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)




    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter +=1

    def choose_action(self, observation):
        # increases its dimension by 1
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.action_space)
        return action

    def learn(self):
        # check to replace target parametrs
        if self.target_update_counter % self.target_update == 0:
            self.sess.run(self.target_replace_op)
            print('\n target_params_replaced \n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run([self._train_op, self.loss], feed_dict ={
                                                                self.s: batch_memory[:, :self.obs_space],
                                                                self.a: batch_memory[:, self.obs_space],
                                                                self.r: batch_memory[:, self.obs_space+1],
                                                                self.s_: batch_memory[:, -self.obs_space:]}
        )

        self.cost_his.append(cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.target_update_counter +=1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':

    env = gym.make('highway-v0')
    agent = DeepQNetwork(env.action_space.n, 25, output_graph=True)

    step = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()

        print("Episode number %s" %(episode))
        while True:
            env.render()
            state = np.array(state).ravel()


            action = agent.choose_action(state)
            print(action)
            state_, reward, done,_ = env.step(action)

            state_ = np.array(state_).ravel()


            agent.store_transition(state, action, reward, state_)
            if (step > 200) and (step % 5 == 0):
                agent.learn()

            state = state_

            if done:
                break
            step += 1

    agent.plot_cost()