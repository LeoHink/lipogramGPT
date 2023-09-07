import torch 
import torch.nn as nn
import pickle 
import pandas as pd
import ast
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter

###### Necissary NLTK Downlaods ########

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

########## Helper Functions ##########

def get_pos_tags(text):
    # Define POS categories of interest
    pos_categories = {
        'Nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
        'Verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'Adjectives': ['JJ', 'JJR', 'JJS'],
        'Adverbs': ['RB', 'RBR', 'RBS']
    }
    pos_tagged_tokens = nltk.pos_tag(word_tokenize(text))

    pos_counts = defaultdict(list)

    # Count POS tags
    pos_counts_temp = Counter()
    for word, tag in  pos_tagged_tokens:
        pos_counts_temp[tag] += 1
    for pos_category, pos_tags in pos_categories.items():
        pos_counts[pos_category].append(sum(pos_counts_temp[tag] for tag in pos_tags))
    
    return pos_counts

def count_u(string):
    """Count the occurrences of 'u' in a string"""
    return string.count('u')


########## Environment Class ##########
 
class GenerationEnv3():
    def __init__(self, unique_words, data, female_set, male_set, decoder, diversity_threshold, max_new_tokens = 500, device = "cpu"):
        self.word_set = unique_words
        self.episode_reward = 0
        self.data = data # encoded torch tensor
        self.device = device
        self.decoder = decoder
        self.diversity_threshold = diversity_threshold
        
        self.init_prompt_len = 128
        self.max_new_tokens = max_new_tokens
        self.activate_rewards = set([8, 2, 12])
        self.verb_threshold = 2
        self.noun_threshold = 3
        self.adj_threshold = 1
        self.adv_threshold = 1
        self.female_set = female_set
        self.male_set = male_set

    def reset(self):
        length = self.init_prompt_len #block size turn into hyperparameter in production (fixed because value network is trained on 128)
        start_idx = torch.randint(len(self.data)-length, (1,)).item()
        end_idx = start_idx+length
        prompt = self.data[start_idx:end_idx]
        prompt = prompt[None, :].to(self.device)
        return prompt #prompt is the initial state

    def reward_function(self, context, action, is_done):
        reward = 0
        if is_done:
            relevant_context = context[0][128:]
            text = self.decoder(relevant_context.tolist())
            u_count = count_u(text)
            unq_words_sent = set(text.split())
            if u_count > 10:
                reward -= 2
            else:
                pos_counts = get_pos_tags(text)
                if pos_counts['Verbs'][0] > self.verb_threshold:
                    reward += 1
                if pos_counts['Nouns'][0] > self.noun_threshold:
                    reward += 1
                if pos_counts['Adjectives'][0] > self.adj_threshold:
                    reward += 1
                if pos_counts['Adverbs'][0] > self.adv_threshold:
                    reward += 1
                if len(unq_words_sent.intersection(self.word_set))/len(text.split()) > 0.66:
                    reward += 1
                # this reward here is an exploit 0/0 or 1/1 will give very high lexical diversity
                if len(unq_words_sent)/len(text.split()) > self.diversity_threshold:
                    reward += 1
        return reward


    def step(self, context, action):
        done = False
        context_new = torch.cat((context, action), dim = 1)
        context_len = context_new.shape[1] - self.init_prompt_len
        if context_len == self.max_new_tokens:
            done = True
            reward = self.reward_function(context_new, action, is_done = done)
        else:
            reward = self.reward_function(context_new, action, is_done = done)

        return context_new, reward, done