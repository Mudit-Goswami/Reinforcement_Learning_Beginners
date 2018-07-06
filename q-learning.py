import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.99153
y = 0.95
num_episodes = 2000

rList = []
for i in range(num_episodes):
 s = env.reset()
 rAll = 0
 d = False
 j = 0
 while j<99:
  j=j+1
  a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
  s1,r,d,_ = env.step(a)
  Q[s,a] = Q[s,a] + learning_rate*(r + y*np.max(Q[s1,:]) - Q[s,a] )
  rAll = rAll + r
  s = s1 
  if d == True:
   break
 rList.append(rAll)

print("Score over Time" + str(sum(rList)/num_episodes))

print("Final Q-Table Values")

print("==================")

print(Q)
