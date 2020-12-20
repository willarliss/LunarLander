import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.base import Agent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
    
        super(Actor, self).__init__()
        
        h1, h2 = 400, 300
        self.max_action = max_action 

        self.layer_1 = nn.Linear(state_dim, h1) 
        self.layer_2 = nn.Linear(h1, h2) 
        self.layer_3 = nn.Linear(h2, action_dim)

    def forward(self, state):

        X = F.relu(self.layer_1(state))
        X = F.relu(self.layer_2(X))
        X = torch.tanh(self.layer_3(X)) 

        return X * self.max_action



class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()
        
        h1, h2 = 400, 300

        self.layer_1 = nn.Linear(state_dim + action_dim, h1)
        self.layer_2 = nn.Linear(h1, h2)
        self.layer_3 = nn.Linear(h2, 1)

    def forward(self, state, action):

        Xu = torch.cat([state, action], axis=1)

        X = F.relu(self.layer_1(Xu))
        X = F.relu(self.layer_2(X))
        X = self.layer_3(X)

        return X


    
class DDPG(Agent):

    def __init__(self, state_dim, action_dim, max_action, mem_size=1e6, eta=1e-3):
        
        super(DDPG, self).__init__()

        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=eta)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=eta)

        self.max_action = max_action
        
        self.mem_size = mem_size
        self.reset_buffer()
        
    def train(self, iterations, batch_size=100, gamma=0.99, policy_noise=0.2, noise_clip=0.5):

        for i in np.arange(iterations):

            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
            
            state = torch.FloatTensor(batch_states).to(DEVICE)
            next_state = torch.FloatTensor(batch_next_states).to(DEVICE)
            action = torch.FloatTensor(batch_actions).to(DEVICE)
            reward = torch.FloatTensor(batch_rewards).to(DEVICE)
            done = torch.FloatTensor(batch_dones).to(DEVICE)

            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip) 
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q =  self.critic_target(next_state, next_action)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()       
            current_Q = self.critic(state, action)
            
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(param.data)
        
    def save(self, filename, directory):

        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')
        
    def load(self, filename, directory):
    
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))


         