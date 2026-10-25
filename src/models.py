"""Reusable heads. BiLSTMAttention is the C2 head (also used on top of a frozen PLM in D2)."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """alpha_i = softmax(v^T tanh(W h_i)),  c = sum_i alpha_i h_i  (padding is masked out)."""

    def __init__(self, dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or dim
        self.W = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class BiLSTMAttention(nn.Module):
    """x: (batch, time, in_dim) float, mask: (batch, time) bool, right-padded."""

    def __init__(self, in_dim, hidden=128, n_classes=2, dropout=0.3, layers=1):
        super().__init__()
        self.drop_in = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.pool = AttentionPooling(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, mask, return_attention=False):
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(self.drop_in(x), lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        context, alpha = self.pool(h, mask)
        logits = self.out(self.drop(context))
        return (logits, alpha) if return_attention else logits
