import gym
import tensorflow as tf
import numpy as np 
import random
from collections import deque
import matplotlib.pyplot as plt


# Hyperparameters
EPISODE = 2000     # Episode limitation
STEP = 500      # Step limitation in an episode
TEST = 3       # The number of experiment test every 100 episode

init_epsilon = 0.5
final_epsilon = 0.01

gamma = 0.9     # discount factor for target Q
replay_size = 10000     # experience replay buffer size
batch_size = 64     # size of minibatch



class DQN():
    def __init__(self, env):    ## 初始化
        ## init experience replay
        self.replay_buffer = deque()
        ## init some parameters
        self.time_step = 0
        self.epsilon = init_epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        

    def create_Q_network(self):     ##创建Q网络
        W1 = tf.Variable(tf.random_normal([self.state_dim, 20]), name='weight1')
        b1 = tf.Variable(tf.zeros([1, 20]) + 0.01, name='biase1')
        W2 = tf.Variable(tf.random_normal([20, 20]), name='weight2')
        b2 = tf.Variable(tf.zeros([1, 20]) + 0.01, name='biase2')
        W3 = tf.Variable(tf.random_normal([20, self.action_dim]), name='weight3')
        b3 = tf.Variable(tf.zeros([1, self.action_dim]) + 0.01, name='biase3')

        self.state_input = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32)

        h_layer1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h_layer = tf.nn.relu(tf.matmul(h_layer1, W2) + b2)

        self.Q_value = tf.matmul(h_layer, W3) + b3


    def create_training_method(self):       ##创建训练方法
        self.action_input = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32)
        self.y_input = tf.placeholder(shape=[None], dtype=tf.float32)

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), axis=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input-Q_action))
        self.train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)


    def store_transition(self, state, action, reward, next_state, done):    ##存储信息
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > replay_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > batch_size:
            self.train_Q_network()


    def train_Q_network(self):      ##训练网络
        self.time_step += 1

        ## step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        ## step 2: calculate y
        y_batch = []
        Q_value_batch = self.sess.run(self.Q_value, feed_dict = {self.state_input: next_state_batch})
        for i in range(batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + gamma*np.max(Q_value_batch[i]))

        self.sess.run(self.train, feed_dict={
            self.y_input: y_batch, 
            self.action_input: action_batch,
            self.state_input: state_batch
            })


    def egreedy_action(self, state):    ##输出带随机的动作
        Q_value_ = self.sess.run(self.Q_value, feed_dict = {self.state_input: state.reshape(-1, self.state_dim)})[0]
        if self.epsilon > final_epsilon:
            self.epsilon -= (init_epsilon-final_epsilon) / 10000
        
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return np.argmax(Q_value_)
        

    def action(self, state):    ##输出动作
        return np.argmax(self.sess.run(self.Q_value, feed_dict = {self.state_input: state.reshape(-1, self.state_dim)})[0])






def main():
    # initialize OpenAI Gym env and DQN agent
    env = gym.make('CartPole-v0')
    agent = DQN(env)

    # 记录得分
    score = np.zeros((100, 2))
    k = 0

    for episode in range(EPISODE):
        ## initialize task
        state = env.reset()
        ## Train
        for step in range(STEP):
            action = agent.egreedy_action(state.reshape(-1, agent.state_dim))    ##e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            ## define reward for agent
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # Test every 20 episodes
        if episode % 20 == 0:
            score[k, 0] = episode 

            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)    ## direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                ave_reward = total_reward / (i+1)
                score[k, 1] = ave_reward
                print('episode:{}, Evaluation Ave Reward:{}'.format(episode, ave_reward))
                if ave_reward >= 200:
                    break

            k += 1

    plt.plot(score[:, 0], score[:, 1])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    main()







