import gym
import time
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import os



def simulate(env, n_est=10, n_epochs=10000, gamma=0.9, alpha=0.01, 
             epsilon=0.95, epsilon_decay=0.999, epsilon_min=0.05,
             memory_size=1000, batch_size=100):

    states = np.zeros((1,env.observation_space.shape[0]))
    q_values = np.zeros((1,env.action_space.n))
    
    mod = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=n_est,
            )
        )
    
    for r in np.arange(n_epochs):
    
            state_0 = env.reset()
            done = False
            
            while not done:
                # env.render()
                
                try:
                    _, X, _, y = train_test_split(states, q_values, test_size=batch_size, shuffle=True)
                    mod.fit(X, y)
                except ValueError:
                    mod.fit(states, q_values)
                
                if np.random.random() < epsilon:
                    action_0 = env.action_space.sample()
                    ov = 0
                else:
                    actions_0 = mod.predict([state_0]) 
                    action_0 = np.argmax(actions_0) 
                    ov = np.max(actions_0)
                
                state_1, reward, done, info = env.step(action_0)                
                actions_1 = mod.predict([state_1])
                fv_max = np.max(actions_1)
                
                fv = (1-alpha)*ov + alpha*(reward + gamma*fv_max)
                q_val = np.zeros(env.action_space.n)
                q_val[action_0] = fv
                
                states = np.append(states, [state_0], axis=0)
                q_values = np.append(q_values, [q_val], axis=0)
                
                states = states[-memory_size:]
                q_values = q_values[-memory_size:]                    
                state_0 = state_1
            
            epsilon = np.max([epsilon_min, epsilon*epsilon_decay])
            
            # if reward == 100:
            #     print('-', r+1, 'success')
            # elif reward == -100:
            #     print(' ', r+1, 'failure')
            
    return (states, q_values, mod)
   


    
if __name__ == '__main__': 
    environment = gym.make('LunarLander-v2') 
    model = simulate(
        environment, 
        n_epochs=100, 
        memory_size=1000,
        batch_size=100,
        gamma=0.8,
        )
    





