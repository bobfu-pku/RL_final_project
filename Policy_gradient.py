import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
#env = env.unwrapped
#env.seed(1)

state_size = 4
action_size = env.action_space.n

max_episodes = 1000d
learning_rate = 0.01
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards

with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
    
    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = input_,
                                                num_outputs = 10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                num_outputs = action_size,
                                                activation_fn= tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                num_outputs = action_size,
                                                activation_fn= None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 
        
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        

max_iternum=200     
total_rewards = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]
goal_last_iternum=10
goal_iternum_stack=np.zeros(goal_last_iternum)
goal=195
Recorder=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        Reward=0
           
        for t in range(max_iternum):
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel()) #依据分布选择动作
            new_state, reward, done, info = env.step(action)
            episode_states.append(state)

            action_ = np.zeros(action_size)
            action_[action] = 1
            
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            Reward+=np.power(gamma,t)*reward
            
            if done or t==max_iternum:
                discounted_episode_rewards= discount_and_normalize_rewards(episode_rewards)
                
                goal_iternum_stack= np.hstack((goal_iternum_stack[1:], [t+1]))   #水平连接
                mean=np.sum(goal_iternum_stack)*1.0/goal_last_iternum
                if episode%20==0:
                    Recorder=Recorder+[mean]
                print("episode: %d, timestep: %d, mean:%f"%(episode,t+1,mean))      
                if mean>goal:
                    break
                #训练
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards 
                                                                })
                episode_states, episode_actions, episode_rewards = [],[],[]
                break
            state=new_state
            
X=np.linspace(0,(len(Recorder)-1)*20,len(Recorder))
plt.figure(figsize=(8,4)) #创建绘图对象
plt.plot(X,Recorder,"b--",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("Episode") #X轴标签
plt.ylabel("Mean")  #Y轴标签
#plt.title("Episode mean for sigmoid decrease: exp=2,sigma=2000") #图标题
plt.show()  #显示图
