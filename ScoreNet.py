import math
import torch
import torch.nn.functional as F
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) /
                          math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, dim_T, num_heads, num_inds, ln=False, gn=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(dim_T, dim_out)
        )
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        if gn:
            self.norm = nn.GroupNorm(16, dim_out)  # groups:16

    def forward(self, X, time_emb):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X) + \
            self.mlp(time_emb)[:, :, None].transpose(1, 2)
        # torch.Size([5, 64, 256]) + torch.Size([5, 1, 256])
        H = H if getattr(self, 'norm', None) is None else self.norm(
            H.transpose(1, 2)).transpose(1, 2)

        return self.mab1(X, H)  # torch.Size([5, 784, 256])



    

class ScoreNet_temb2(nn.Module):
    def __init__(
            self,
            dim_input=4,
            dim_output=1,
            num_inds=64,
            dim_hidden=256,
            dim_time=32,
            num_heads=4,
            ln=False,
            gn=False,
            dropout=0.1
    ):
        super(ScoreNet_temb2, self).__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim_time),
            nn.Linear(dim_time, dim_time * 4),
            Swish(),
            nn.Linear(dim_time * 4, dim_time)
        )
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
            ISAB(dim_hidden, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
            ISAB(dim_hidden, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
            ISAB(dim_hidden, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
            ISAB(dim_hidden, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
            ISAB(dim_hidden, dim_hidden, dim_time,
                 num_heads, num_inds, ln=ln, gn=gn),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X, time):

        t = self.time_mlp(time)  # torch.Size([5, 32])
        for layer in self.enc:
            X = layer(X, t)

        return self.dec(X)


if __name__ == '__main__':

    m=TimeEmbedding(32)

    a=torch.FloatTensor([0.015,0.01])
    print(m(a))