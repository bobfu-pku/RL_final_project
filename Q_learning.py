import gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('CartPole-v0')
#总共实现1000次场景，每次场景的最大时间步长为200
episode_num=10000
max_stepnum=200   
epsilon=0.1
alpha=0.2
gamma=0.99999
#获胜条件为最近10场的平均分高于195
goal_average_step=195   
goal_last_iternum=100
goal_iternum_stack=np.zeros(goal_last_iternum)


#由于Q-Learning为基于表格的强化学习，所以必须对状态空间离散化

def creatbins(statemin,statemax,num):  
    return np.linspace(statemin,statemax,num+1)[1:-1]    #注意，这里得到了num块区间，同时为了保证状态空间，首尾去掉
	
def dis_state(observation):      #通过np.digitize()函数将连续状态转变为离散状态
    cart_pos,cart_v,pole_angle,pole_v=observation
    state=[np.digitize(cart_pos,bins=creatbins(-1*statemax[0],statemax[0],num)),
           np.digitize(cart_v,bins=creatbins(-1*statemax[1],statemax[1],num)),
           np.digitize(pole_angle,bins=creatbins(-1*statemax[2],statemax[2],num)),
           np.digitize(pole_v,bins=creatbins(-1*statemax[3],statemax[3],num))]
    
	#离散化后的状态为 num*num*num*num 大小的，所以用num进制表示，使得状态和索引一一对应
    k=0
    for i in range(4):
    	k+=state[i]*np.power(num,i)
    return k

#状态空间离散化后做Q—table
num=20
statemax=[4.8,10,0.5,10]
Q_table=np.random.uniform(low=-1,high=1,size=(num**4,env.action_space.n))  


#定义动作选择策略
def get_action(epsilon,state,Q_table,episode):
    a=np.random.uniform(0,1)
    if a>epsilon:
        return np.argmax(Q_table[state])     #1-epsilon 的概率选择最优动作
    else:
        return np.random.choice([0,1])		 #余下随机选择一个动作

"""
#对epsilon-greedy引入一个sigmoid类型epsilon递减函数
sigmoid_sigma=2000
sigmoid_exp=2

def get_action(epsilon,state,Q_table,episode):
    a=np.random.uniform(0,1)
    act_eps=epsilon*1.0/(1+np.exp(sigmoid_exp*(episode-sigmoid_sigma)))
    if a>act_eps:
        return np.argmax(Q_table[state])     #1-epsilon 的概率选择最优动作
    else:
        return np.random.choice([0,1])		 #余下随机选择一个动作
"""
    
def Q_step(epsilon,alpha,gamma,episode):    #单步Q过程
    #env.render()
    observation=env.reset()    #初始状态
    state=dis_state(observation)
    Reward=0
    for t in range(max_stepnum):
        #action=get_action(epsilon,state,Q_table)
        action=get_action(epsilon,state,Q_table,episode)
        observation,reward,done,info=env.step(action)
        Reward+=np.power(gamma,t)*reward
        next_state=dis_state(observation)
		
        
        Q_table[state,action]+=alpha*(reward+gamma*np.max(Q_table[next_state])-Q_table[state,action])
        state=next_state
		
        if done:
            return Reward,t
        """
        
        #添加失败惩罚
        Q_table[state,action]+=alpha*(reward+gamma*np.max(Q_table[next_state])-Q_table[state,action])
        if done:
            reward=-10
            Q_table[state,action]+=alpha*reward
            return Reward,t
        state=next_state
        """

"""
环境描述:
Oservation:
    Cart Positio: min=-4.8,max=4.8
    Cart Velocity: min=-inf, max=inf
    Pole Angle: min=-0.418, max=0.418
    Pole Velocity At Tip: min=-inf, max=inf
"""        
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
num=20 #将每个连续状态分成num种离散特征
statemax=[4.8,10,0.5,10]    #给出状态的边界，状态对称，虽然有无穷的情况，但是实际运行几乎在10之内

Recorder=[]
e=0
for episode in range(episode_num):
    Reward,t=Q_step(epsilon,alpha,gamma,episode)
    goal_iternum_stack= np.hstack((goal_iternum_stack[1:], [Reward]))   #水平连接
    mean=np.sum(goal_iternum_stack)*1.0/goal_last_iternum
    print("%d Episode finished after %f time steps: mean %f"%(episode,t+1,mean))
    if np.mod(episode,20)==0:
        Recorder+=[mean]
    if mean >= goal_average_step:
        print("train sucessfully!")
        break


X=np.linspace(0,(len(Recorder)-1)*20,len(Recorder))
plt.figure(figsize=(8,4)) #创建绘图对象
plt.plot(X,Recorder,"b--",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("Episode") #X轴标签
plt.ylabel("Mean")  #Y轴标签
#plt.title("Episode mean for sigmoid decrease: exp=2,sigma=2000") #图标题
plt.show()  #显示图


    
    

	