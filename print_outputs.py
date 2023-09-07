import os 
import torch 
from langmodel import LanguageModel
from GenEnv import GenerationEnv3
import datetime
import pickle
import pandas as pd


################## Data Imports ########################
df = pd.read_csv('datasets/generated_data.csv')

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

with open('encoded_data_tensor.pkl', 'rb') as file:
    loaded_tensor = pickle.load(file)

data = loaded_tensor

with open('datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary Size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])

########################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LanguageModel().to(device)

env = GenerationEnv3(unique_words, data, avg_words, avg_a_count, decode, 0.5, device=device)

########################################################

def print_outputs(weights_dir):
    output_list = []
    context = env.reset()
    current_date = datetime.datetime.now()
    date_string = current_date.strftime('%Y_%m_%d')
    for file in os.listdir(weights_dir):
        if file.endswith(".pth"): 
            weights_path = os.path.join(weights_dir, file)
            model.load_state_dict(torch.load(weights_path))
            
            output = model.generate(context, max_new_tokens=500)[0].tolist()

            decoded_output = decode(output)
            print(f"{file} output: {decoded_output}")

            # save to list
            output_list.append(decoded_output)
    
    with open(f'{date_string}_encoded_outputs.pkl', 'wb')as output_file:
        pickle.dump(output_list, output_file)
    
    return output_list
