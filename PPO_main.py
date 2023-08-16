import torch 
import numpy as np
import pandas as pd
import pickle
import ast
from torch.distributions import Categorical
from torch import nn
from langmodel import LanguageModel
from torch.nn import functional as F
from value_network import ValueNetwork
import numpy as np
import datetime as datetime
import time
from GenEnv import GenerationEnv3
import ast

#### THIS IS THE LAMBA LABS VERSION OF PPO ####

################## Data Imports ########################
df = pd.read_csv('generated_data.csv')

avg_words = df['word_count'].mean()
avg_a_count = df['a_count'].mean()

df['output'] = df['output'].apply(ast.literal_eval)

df['output_length'] = df['output'].apply(lambda x: len(x))
average_length = df['output_length'].mean()

def load_set_from_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data

unique_words = load_set_from_file('shakespeare_word_set.pkl')

data = load_set_from_file('encoded_data_tensor.pkl')

female_characs = load_set_from_file('female_characs_set.pkl')

male_characs = load_set_from_file('male_characs_set.pkl')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary Size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])

################## Globl Variables ########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = GenerationEnv3(unique_words, data, avg_words, avg_a_count, decode, 0.5, device=device)

################### Hyperparameters #######################

name = "RLHF"

log_interval = 20           # print avg reward in the interval
max_episodes = 1_000        # max training episodes
max_timesteps = 500         # max timesteps in one episode

update_timestep = 2000      # update policy every n timesteps
vocab_size = 65
embed_dim = 256
hidden_dim = 512
solved_reward = 150        # stop training if avg_reward > solved_reward
block_size = 128

lr = 0.0006 #I will reduce the learning rate
betas = (0.9, 0.999)        # same as the default Adam betas
gamma = 1                   # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None

################### Helper Functions #######################
def save_training_hist(dict):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(f'training_log_{date}.pkl', 'wb') as f:
        pickle.dump(dict, f)


def save_model(ppo, name, epoch):
    # Get the current date
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    # Save the model
    torch.save(ppo.actor.state_dict(), 'PPO_{}_{}_Epoch_{}.pth'.format(name, date, epoch))


def freeze_weights(model):
  
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze the parameters you are interested in
    for name, param in model.named_parameters():
        if 'blocks.3' in name or name in ['ln_f.weight', 'ln_f.bias', 'lm_head.weight', 'lm_head.bias']:
            param.requires_grad = True

    return model

#############################################################


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO2:
    def __init__(self, vocab_size, embed_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip, block_size):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.block_size = block_size


        self.actor = LanguageModel().to(device) # our policy given by transformer
        self.actor.load_state_dict(torch.load('shakespeare_base.pth', map_location=torch.device(device))) #loading pre-trained weights
        self.actor = freeze_weights(self.actor) # freezing all but the last transformer block and outputs (softmax) layer
        
        self.critic = ValueNetwork(vocab_size, embed_dim, hidden_dim).to(device)
        self.act_optimizer = torch.optim.Adam(self.actor.parameters(),
                                              lr=lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                              lr=lr, betas=betas) # our value network to calculate advantage

        self.policy_old = LanguageModel().to(device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def act(self, state, memory):
        state = state[:, -self.block_size:]
        logits, _ = self.actor(state)
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        log_probs = F.log_softmax(logits, dim=-1) # log_probs used in PPO update
        # sample from the distribution
        action = torch.multinomial(probs, num_samples=1) # (B, 1)
       
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_probs.squeeze(0)[action]) # B

        return action # tensor of dim (1, 1)

    def update(self, memory):
            # Monte Carlo estimate of state rewards
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward) # insert at front to regain original order

            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # Convert list to tensor
            old_states = torch.stack(memory.states).to(device).detach().squeeze(1)
            old_actions = torch.stack(memory.actions).to(device).detach().squeeze(1)
            old_logprobs = torch.stack(memory.logprobs).to(device).detach().squeeze(1)

            # Optimization step
            for _ in range(self.K_epochs):
                # Get policy logits
                logits, _ = self.actor(old_states)
                logits = logits[:, -1, :] # only care about the last logits
                # Get probabilities from logits
                probs = F.softmax(logits, dim=-1)
                
                # Get log probabilities
                logprobs = F.log_softmax(logits, dim=-1)
                
                # Get the logprob of the taken action
                logprobs = torch.gather(logprobs, 1, old_actions)

                # Calculating the entropy
                dist_entropy = -(probs * logprobs).sum(-1).mean()

                # Evaluating old values
                state_values = self.critic(old_states)

                # Calculating the policy loss
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

                # Calculating the value loss
                value_loss = 0.5 * self.MseLoss(state_values, rewards)


                # Taking a gradient step for policy network
                self.act_optimizer.zero_grad()
                policy_loss.mean().backward() # get average policy loss across the batch dimension
                self.act_optimizer.step()

                # Taking a gradient step for value network
                self.critic_optimizer.zero_grad()
                value_loss.mean().backward()
                self.critic_optimizer.step()

def main():

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO2(vocab_size, embed_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip, block_size)
    print(f"The Learning Rate is: {lr}, The Betas for Adam Optimizer are: {betas}")

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    training_dict = {'reward_history': [], 'cumulative_reward_history': [], 'time_history': []}
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        episode_rewards = 0
        start_time = time.time()
        old_weights = ppo.actor.state_dict()
        for t in range(max_timesteps):
            time_step +=1
            # running policy_old:

            action = ppo.act(state, memory)

            state, reward, done = env.step(state, action)
            state.to(device)
            
            episode_rewards += reward
            # saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                # print(memory.states)
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward

            if done:
                break
                
        training_dict['cumulative_reward_history'].append(running_reward)
        training_dict['reward_history'].append(episode_rewards)

        avg_length += t

        # stop training if we reach a high level of reward 
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            torch.save(ppo.actor.state_dict(), 'PPO_{}.pth'.format(name)) # i do like this here through. 
            time_to_finish = time.time() - sum(training_dict['time_history'])

            break
        if i_episode == max_episodes+1:
            time_to_finish = time.time() - sum(training_dict['time_history'])
            break
        # logging
        if i_episode % log_interval == 0: # this I also like
            avg_length = avg_length/log_interval
            update_interval = time.time() - start_time
            training_dict['time_history'].append(update_interval)
            running_reward = running_reward/log_interval
            save_model(ppo, name, i_episode)
            save_training_hist(training_dict)

            print('Episode {} \t num tokens generated: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            
            
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()