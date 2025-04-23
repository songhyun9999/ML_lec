import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V Linear
        self.q_linear = nn.Linear(embed_dim,embed_dim)
        self.k_linear = nn.Linear(embed_dim,embed_dim)
        self.v_linear = nn.Linear(embed_dim,embed_dim)

        self.out_proj = nn.Linear(embed_dim,embed_dim)

        self.scale = embed_dim ** 0.5


    def forward(self,x):
        B,S,D = x.size()     # (batch_size, seq_len, embedding_dim)
        Q = self.q_linear(x) # (batch_size, seq_len, embedding_dim)
        K = self.k_linear(x) # (batch_size, seq_len, embedding_dim)
        V = self.v_linear(x) # (batch_size, seq_len, embedding_dim)
        # (B, S, D) -> (B, num_heads, S, head_num)
        Q = Q.view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(B,S,self.num_heads,self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(Q,K.transpose(-2,-1)) / self.scale
        attention_weights = F.softmax(attention_scores,dim=-1)
        context = torch.matmul(attention_weights,V)

        output = self.out_proj(context)

        return output, attention_weights


batch_size = 1
seq_len = 4
embedding_dim = 8
num_heads = 1

x = torch.randn(batch_size,seq_len,embedding_dim)

sAttention = MultiHeadAttention(embedding_dim,num_heads)
output,weights = sAttention(x)
print('output shape : ',output.shape)
print('attention weight : ',weights)


