import torch 
from torch import nn

########## Hyperparameters ###########

vocab_size = 65
embd_dim = 256
hidden_dim = 512

#####################################

class ValueNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.embedding(state)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # We only want the final output of the LSTM
        return x