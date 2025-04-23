import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerPart.day1.transformerEx2 import num_heads


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

        attention_scores = torch.matmul(Q,K.transpose(-2,-1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(attention_scores,dim=-1)
        context = torch.matmul(attention_weights,V)
        context = context.transpose(1,2).contiguous().view(B,S,self.embed_dim) # (B,S,D)

        output = self.out_proj(context)

        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim,ff_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(ff_dim,embed_dim)

    def forward(self,x):
        x = self.dropout(F.relu(self.linear1(x)))
        y = self.linear2(x)

        return y

class TransformerEncoder(nn.Module):
    def __init__(self,embed_dim,num_heads,ff_dim,dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim,num_heads)
        self.att_layer_norms = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim,ff_dim,dropout)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        attn_output, attn_weights = self.attn(x)
        x = self.att_layer_norms((x + self.dropout(attn_output)))
        ffn_output = self.ffn(x)
        x + self.dropout(ffn_output)
        y = self.ffn_layer_norm(x + self.dropout(ffn_output))

        return y, attn_weights


batch_size = 2
seq_len = 10
embedding_dim = 512
num_heads = 8
ff_dim = 2048

x = torch.randn(batch_size,seq_len,embedding_dim)
encoder_layer = TransformerEncoder(embed_dim=embedding_dim,num_heads=num_heads,ff_dim=ff_dim)

output,attn_weight = encoder_layer(x)

print('output shape : ',output.shape)
print('attention weight : ',attn_weight)


