import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V Linear
        self.q_linear = nn.Linear(embed_dim,embed_dim)
        self.k_linear = nn.Linear(embed_dim,embed_dim)
        self.v_linear = nn.Linear(embed_dim,embed_dim)

        self.scale = embed_dim ** 0.5


    def forward(self,x):
        Q = self.q_linear(x) # (batch_size, seq_len, embedding_dim)
        K = self.k_linear(x) # (batch_size, seq_len, embedding_dim)
        V = self.v_linear(x) # (batch_size, seq_len, embedding_dim)

        attention_scores = torch.matmul(Q,K.transpose(-2,-1)) / self.scale
        attention_weights = F.softmax(attention_scores,dim=-1)
        output = torch.matmul(attention_weights,V)

        return output, attention_weights


batch_size = 1
seq_len = 4
embedding_dim = 8

x = torch.randn(batch_size,seq_len,embedding_dim)

sAttention = SelfAttentionNN(embedding_dim)
output,weights = sAttention(x)
print('output shape : ',output.shape)
print('attention weight : ',weights)


