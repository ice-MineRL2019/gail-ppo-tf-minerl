import numpy as np
import matplotlib.pyplot as plt
import random

class state_action_newstate:
    def __init__(self,observation,action,new_observation):
        self.state=observation
        self.action=action
        self.new_state=new_observation

if __name__ == "__main__":
    observation_new=np.zeros([84,84])
    state_action_set=set()
    env=None
    env.reset()
    #plt.imshow(observation,cmap='gray')
    #plt.show()
    print(env.observation_space)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    for t in range(5000):
        env.render() #刷新环境
        #print(observation.shape)
        #action1   action2
        observation_original=observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        new_object=state_action_newstate(observation_original,action,observation)
        state_action_set.add(new_object)
        # if done:
        #     print("Episode finished after {} timesteps".format(t + 1))
        #     break
    print(np.size(state_action_set))