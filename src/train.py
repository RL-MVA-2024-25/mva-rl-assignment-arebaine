from gymnasium.wrappers import TimeLimit
# from fast_env import FastHIVPatient
from dqn import DQN
from env_hiv import HIVPatient
from ReplayBuffer import ReplayBuffer
import torch
from torch import nn
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
class ProjectAgent:
    def act(self, observation, use_random=False):
        return self.model(torch.tensor(observation, dtype=torch.float).unsqueeze(0)).argmax().item()

    def save(self, path):
        torch.save(self.model.state_dict(), "models/model.pt")

    def load(self):
        self.model = DQN()
        self.model.load_state_dict(torch.load("models/dict_transitoire.pt", weights_only=True))

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, max_episode, use_random=False):
        env = TimeLimit(env=HIVPatient(domain_randomization=use_random), max_episode_steps=200) 
        self.path_to_save = "models/dict.pt"
        self.memory = ReplayBuffer(capacity=100000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DQN()
        self.target = DQN()
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.update_target_network_every = 100
        self.gamma = 0.95
        self.epsilon_init = 1
        self.epsilon_min = 0.05
        self.epsilon_stop = 10000
        self.epsilon_delay = 20
        self.epsilon_step = (self.epsilon_init-self.epsilon_min)/self.epsilon_stop
        self.batch_size = 100
        self.nb_gradient_steps = 5
        epsilon = self.epsilon_init
        state, _ = env.reset()
        save_rewards = []
        step = 0
        episode = 0
        episode_cum_reward = 0
        print("Begin training...")
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if step % self.update_target_network_every == 0: 
                self.target.load_state_dict(self.model.state_dict())
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                torch.save(self.model.state_dict(), "models/dict_transitoire.pt")
                if episode%10==0:
                    torch.save(self.model.state_dict(), "models/checkpoint.pt")
                # Monitoring
                save_rewards.append(episode_cum_reward)
                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", buffer size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                        sep='')
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        
        torch.save(self.model.state_dict(), self.path_to_save)
        return episode_cum_reward

if __name__ == '__main__':
    agent = ProjectAgent()
    agent.train(max_episode=10000, use_random=False)



        

                

            
















