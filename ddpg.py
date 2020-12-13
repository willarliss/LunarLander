import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ReplayBuffer(object):

    def __init__(self, max_len=1e6):

        self.ptr = 0
        self.storage = []
        self.max_len = max_len
    
    def add(self, transition):

        if len(self.storage) == self.max_len:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_len

        else:
            self.storage.append(transition)

    def sample(self, batch_size):

        idx = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in idx:
            state, next_state, action, reward, done = self.storage[i]
      
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape((-1,1)),
            np.array(batch_dones).reshape((-1,1)), 
        )



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
        X = torch.tanh(self.layer_3(X)) * self.max_action

        return X



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
        X = (self.layer_3(X))

        return X


class DDPG(object):

    def __init__(self, state_dim, action_dim, max_action, mem_size=1e6, eta=1e-3):

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
        self.replay_buffer = ReplayBuffer(max_len=self.mem_size)

    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)

        return (self.actor(state)
            .cpu()
            .data
            .numpy()
            .flatten()
        )

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
                
    def reset_buffer(self):
        
        self.replay_buffer = ReplayBuffer(max_len=self.mem_size)
        
    def save(self, filename, directory):

        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')
        
    def load(self, filename, directory):
    
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))


         