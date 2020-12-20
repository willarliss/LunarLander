import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ReplayBuffer:

    def __init__(self, max_len=1e6):

        self.ptr = 0
        self.storage = []
        self.max_len = max_len
        self.curr_len = len(self.storage)
    
    def add(self, transition):

        if self.curr_len == self.max_len:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_len

        else:
            self.storage.append(transition)
            
        self.curr_len = len(self.storage)

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



class Agent:

    def __init__(self, *args, mem_size=1e6, eta=1e-3, **kwargs):

        self.mem_size = mem_size
        self.replay_buffer = self.reset_replay_buffer()

    def select_action(self, state, **kwargs):

        state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)

        return (self.actor(state)
            .cpu()
            .data
            .numpy()
            .flatten()
        )

    def reset_replay_buffer(self, inplace=False, size=None, **kwargs):
        
        if size is None:
            size = self.mem_size
            
        if inplace:
            self.replay_buffer = ReplayBuffer(max_len=size)
        else:
            return ReplayBuffer(max_len=size)
        
    def train(self, iterations=100, batch_size=100, gamma=0.99, policy_noise=0.2, noise_clip=0.5, **kwargs):

        pass
    
    def save(self, filename, directory, **kwargs):

        pass
    
    def load(self, filename, directory, **kwargs):
    
        pass


         