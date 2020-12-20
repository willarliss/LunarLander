import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.base import Agent

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Value(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Value, self).__init__()
        
        h1, h2 = 200, 100

        self.layer_1 = nn.Linear(state_dim, h1)
        self.layer_2 = nn.Linear(h1, h2)
        self.layer_3 = nn.Linear(h2, action_dim)

    def forward(self, state):
        
        X = F.relu(self.layer_1(state))
        X = F.relu(self.layer_2(X))
        X = self.layer_3(X)

        return X

    

class DQN(Agent):

    def __init__(self, state_dim, action_dim, mem_size=1e6, eta=1e-3):
        
        super(DQN, self).__init__()
        
        self.value_fn = Value(state_dim, action_dim).to(DEVICE)
        self.value_fn_target = Value(state_dim, action_dim).to(DEVICE)
        self.value_fn_target.load_state_dict(self.value_fn.state_dict())
        
        self.value_fn_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=eta)

        self.mem_size = mem_size
        self.reset_buffer()

    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)
        
        Q_values = (self.value_fn(state)
            .cpu()
            .data
            .numpy()
            .flatten()
        )
        
        return np.argmax(Q_values)
        
    def train(self, iterations, batch_size=100, gamma=0.99, tau=0.0, policy_freq=2):

        for i in np.arange(iterations):

            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
            
            state = torch.FloatTensor(batch_states).to(DEVICE)
            next_state = torch.FloatTensor(batch_next_states).to(DEVICE)
            action = torch.FloatTensor(batch_actions).to(DEVICE)
            reward = torch.FloatTensor(batch_rewards).to(DEVICE)
            done = torch.FloatTensor(batch_dones).to(DEVICE)

            target_Qs = self.value_fn_target(next_state)
            current_Qs = self.value_fn(state)
        
            target_Q = torch.max(target_Qs, dim=1)[0].reshape((-1,1))
            target_Q = reward + ((1-done) * gamma * target_Q).detach()   
 
            current_Q = current_Qs[
                np.arange(current_Qs.shape[0]), 
                action.numpy().flatten(),
                ].reshape((-1,1))
                    
            Q_loss = F.mse_loss(current_Q, target_Q)
            self.value_fn_optimizer.zero_grad()
            Q_loss.backward()
            self.value_fn_optimizer.step()

            if i % policy_freq == 0:

                for param, target_param in zip(self.value_fn.parameters(), self.value_fn_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )
        
    def save(self, filename, directory):
        
        torch.save(self.value_fn.state_dict(), f'{directory}/{filename}_value_fn.pth')
        
    def load(self, filename, directory):
    
        self.value_fn.load_state_dict(torch.load(f'{directory}/{filename}_value_fn.pth'))

        