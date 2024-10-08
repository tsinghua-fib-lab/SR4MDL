import time
import torch
import torch.nn as nn
from math import log
from typing import List


def noam_lambda(d_model, warmup_steps):
    return lambda step: d_model**-0.5 * min((step+1e-12)**-0.5, (step+1e-12) * warmup_steps**-1.5) * (warmup_steps)**0.5


class MLP(nn.Module):
    def __init__(self, *dim_list:List[int], act=nn.ReLU, out_act=nn.ReLU, dropout=0.0, dropout_output=True):
        super(MLP, self).__init__()
        seq = []
        for i, j in zip(dim_list[:-1], dim_list[1:]):
            seq.append(nn.Linear(i, j))
            seq.append(act())
            seq.append(nn.Dropout(p=dropout))
        seq[-2] = out_act()
        if not dropout_output: seq.pop()
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    # batch_first=True
    def __init__(self, d_emb, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_emb, 2, dtype=torch.float) * (-log(10000.0) / d_emb)).unsqueeze(0) # (1, d_emb//2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[-2], :]
        return self.dropout(x)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_emb)
            padding_mask: (batch, seq_len)
        Returns:
            pooled: (batch, d_emb)
        """
        if padding_mask is not None:
            x.masked_fill_(padding_mask.unsqueeze(-1), 0)
        return x.sum(dim=1) / (padding_mask.sum(dim=1, keepdim=True) if padding_mask is not None else x.shape[1])


class AttentionPooling(nn.Module):
    def __init__(self, d_emb):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_emb, 1)
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_emb)
            padding_mask: (batch, seq_len)
        Returns:
            pooled: (batch, d_emb)
        """
        w = self.attention(x) # (batch, seq_len, 1)
        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(-1), -1e9)
        w = torch.softmax(w, dim=1) # (batch, seq_len, 1)
        pooled = (w * x).sum(dim=1) # (batch, d_emb)
        return pooled


class SENet(nn.Module):
    def __init__(self, n_channel, d_emb, reduce=1, use_sigmoid=True, n_SENet_head=1):
        super(SENet, self).__init__()
        self.attention = nn.Linear(d_emb, n_SENet_head)
        self.seq = nn.Sequential(
            nn.Linear(n_channel*n_SENet_head, n_channel*n_SENet_head // reduce),
            nn.ReLU(),
            nn.Linear(n_channel*n_SENet_head // reduce, n_channel),
            nn.Sigmoid() if use_sigmoid else nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channel, d_emb)
        Returns:
            w: (batch, channel)
        """
        w = self.attention(x) # (batch, seq_len, channel, n_SENet_head)
        w = w.mean(dim=1).flatten(-2, -1) # (batch, channel * n_SENet_head)
        w = self.seq(w) # (batch, channel)
        return w


class AttentionPooling2(nn.Module):
    def __init__(self, d_emb, n_head=8):
        super(AttentionPooling2, self).__init__()
        self.attention = nn.Linear(d_emb, n_head)
        self.value = nn.Linear(d_emb, n_head * d_emb)
        self.n_head = n_head
        self.d_emb = d_emb
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_emb)
            padding_mask: (batch, seq_len)
        Returns:
            pooled: (batch, d_emb)
        """
        w = self.attention(x) # (batch, seq_len, n_head)
        v = self.value(x).reshape(*x.shape[:-1], self.n_head, self.d_emb) # (batch, seq_len, n_head, d_emb)
        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(-1), -1e9)
        w = torch.softmax(w, dim=1).unsqueeze(-1) # (batch, seq_len, n_head, 1)
        pooled = (w * v).sum(dim=[1,2]) # (batch, d_emb)
        return pooled


class CLIPLoss(nn.Module):
    def __init__(self, clip_temperature=1.0):
        super(CLIPLoss, self).__init__()
        self.clip_temperature = clip_temperature
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        logits = (x @ y.T) / self.clip_temperature
        labels = torch.arange(logits.shape[0], device=x.device, dtype=torch.long)
        loss = (self.loss(logits, labels) + self.loss(logits.T, labels)) / 2
        return loss.mean()


class Timer:
    def __init__(self):
        self.count = 0
        self.time = 0
        self.start_time = time.time()
    
    def add(self, n):
        self.count += n
        self.time += time.time() - self.start_time
        self.start_time = time.time()
    
    def pop(self, reset=True):
        speed = self.count / self.time
        if reset: self.count = self.time = 0
        return speed


class CycleTimer(Timer):
    def __init__(self, N):
        self.N = N
        self.time = [0] * N
        self.start_time = time.time()
    
    def add(self, idx):
        self.time[idx] += time.time() - self.start_time
        self.start_time = time.time()
    
    def pop(self, reset=True):
        total = sum(self.time)
        ratio = [t / total for t in self.time]
        if reset: self.time = [0] * self.N
        return total, ratio


class NamedTimer(Timer):
    def __init__(self):
        self.time = {}
        self.count = {}
        self.start_time = time.time()
    
    def __str__(self):
        total_time = sum(self.time.values())
        return f'{total_time*1000:.0f}ms' + ' (\n' + '\n'.join(
                   f'  {k:30} : {self.time[k] / total_time:3.0%} = {self.count[k]:5} * {self.time[k] / self.count[k] * 1000 ** 2:5,.0f}us'
                   for k in sorted(self.time, key=self.time.get, reverse=True)
               ) + '\n)'

    def add(self, name):
        if name not in self.time: self.time[name] = self.count[name] = 0
        self.time[name] += time.time() - self.start_time
        self.count[name] += 1
        self.start_time = time.time()
    
    def pop(self, reset=True):
        res = self.time
        if reset: self.time = {}
        return res

    def total(self):
        return sum(self.time.values())
