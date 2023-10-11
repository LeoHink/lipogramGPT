import torch 
import torch.nn as nn
import pickle 
import pandas as pd
import ast
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import re

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

    # Count POS tags at the document and sentence levels
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
    def __init__(self, unique_words, data, decoder, diversity_threshold, max_new_tokens = 500, device = "cpu"):
        self.word_set = unique_words
        self.episode_reward = 0
        self.data = data # encoded torch tensor
        self.device = device
        self.decoder = decoder
        self.diversity_threshold = diversity_threshold
        
        self.init_prompt_len = 128
        self.max_new_tokens = max_new_tokens
        self.verb_threshold = 15
        self.noun_threshold = 27
        self.adj_threshold = 6
        self.adv_threshold = 5
        self.u_threshold = 10

    def reset(self):
        length = 128 #block size turn into hyperparameter in production (fixed because value network is trained on 128)
        start_idx = random_index = torch.randint(len(self.data)-length, (1,)).item()
        end_idx = start_idx+length
        prompt = self.data[start_idx:end_idx]
        prompt = prompt[None, :].to(self.device)
        self.init_prompt_len = prompt.shape[1]
        return prompt #prompt is the initial state

    def reward_function(self, context, action, is_done):
        reward = 0
        if is_done:
            relevant_context = context[0][self.init_prompt_len:]
            text = self.decoder(relevant_context.tolist())
            u_count = count_u(text)
            text_no_punkt = re.sub(r'[^\w\s]', '', text)
            unq_words_set = set(text_no_punkt.split())
            print(f'U Count: {u_count}\n')
            if u_count > self.u_threshold:
                reward -= 2
            else:
                pos_counts = get_pos_tags(text)
                if pos_counts['Verbs'][0] >= self.verb_threshold:
                    print(f"Verbs: {pos_counts['Verbs'][0]}\n")
                    reward += 1
                if pos_counts['Nouns'][0] >= self.noun_threshold:
                    print(f"Nouns: {pos_counts['Nouns'][0]}\n")
                    reward += 1
                if pos_counts['Adjectives'][0] >= self.adj_threshold:
                    print(f"Adj: {pos_counts['Adjectives'][0]}\n")
                    reward += 1
                if pos_counts['Adverbs'][0] >= self.adv_threshold:
                    print(f"Adverbs: {pos_counts['Adverbs'][0]}\n")
                    reward += 1
                if len(unq_words_set.intersection(self.word_set))/len(unq_words_set) >= 0.8:
                    print(f'real word count: {len(unq_words_set.intersection(self.word_set))/len(unq_words_set)}\n')
                    reward += 1
                # this reward here is an exploit 0/0 or 1/1 will give very high lexical diversity
                if len(unq_words_set)/len(text.split()) > self.diversity_threshold:
                    print(f'Lexical Div: {len(unq_words_set)/len(text.split())}')
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