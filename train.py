# ---------------
# Based on Karpathy (2023): https://github.com/karpathy/ng-video-lecture
# ---------------


import torch 
from torch import nn 
import matplotlib.pyplot as plt
from langmodel import LanguageModel

with open('input.txt', 'r') as f:
    text = f.read()

############### Data Loading #####################

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = LanguageModel()

################### hyperparameters #########################

batch_size = 64
block_size = 128
eval_iters = 2_500
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
max_iters = 500
eval_interval = 1
learning_rate = 1e-3


################### helpers #########################

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_model(model, step):
    # Save the model
    torch.save(model.state_dict(), 'transformer_checkpoint_step_{}.pth'.format(step))

def plot_learning_curve(history):
    plt.figure(figsize=(10,6))

    # plot train_loss
    plt.plot(history['train_loss'], label='Training loss')

    # plot val_loss
    plt.plot(history['val_loss'], label='Validation loss')

    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Loss')

    plt.legend()

    plt.show()

    plt.savefig('transformer_training_curve.png')



####################### main ##############################

def main():

    model.train()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # AdamW Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': []}
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            history['train_loss'].append(losses['train'])
            history['val_loss'].append(losses['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            save_model(model, iter)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model at end of training
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    

if __name__ == '__main__':
    main()