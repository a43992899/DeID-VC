import torch
import torch.nn as nn


class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
        
        
    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized


class D_VECTOR_tune(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR_tune, self).__init__()
        self.d_vector = D_VECTOR(num_layers, dim_input, dim_cell, dim_emb)
        self.proj = nn.Linear(dim_emb, dim_emb)
        
        
    def forward(self, x):
        with torch.no_grad():
            embed = self.d_vector(x)
        embed = self.proj(embed)
        return embed


class Proj(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, in_dim=256, out_dim=256):
        super(Proj, self).__init__()
        self.proj0 = nn.Linear(in_dim, 512)
        self.sigmoid = nn.Sigmoid()
        self.proj1 = nn.Linear(512, out_dim)
              
    def forward(self, x):
        x = self.proj0(x)
        x = self.proj1(x)
        x = self.sigmoid(x)
        return x
