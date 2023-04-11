import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

OPS = {
    'identity': lambda d_model: Identity(d_model),
    'conv': lambda d_model: Conv(d_model),
    'attention': lambda d_model: Attention(d_model),
    'rnn': lambda d_model: RNN(d_model),
    'ffn': lambda d_model: FFN(d_model),
    'zero': lambda d_model: Zero(d_model)
}

class Zero(nn.Module):
    def __init__(self, d_model):
        super(Zero, self).__init__()
    def forward(self, x, masks, lengths):
        return torch.mul(x, 0)


class Conv(nn.Module):
    def __init__(self, d_model):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model, affine=True)
        )
        # self.batchnm = nn.BatchNorm1d(d_model, affine=True)
        # self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)

    def forward(self, x, masks, lengths):
        x = self.op(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class FFN(nn.Module):

  def __init__(self, d_model):
      super(FFN, self).__init__()
      self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
      self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x, masks, lengths):
      x = self.layer_norm(x + self.ffn(x))
      return x


class Identity(nn.Module):
  def __init__(self, d_model):
      super(Identity, self).__init__()
  def forward(self, x, masks, lengths):
      return x


class Attention(nn.Module):
    def __init__(self, in_feature, num_head=4, dropout=0.1):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, x, attn_mask, lengths):
        batch_size = x.size(0)
        res = x
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            batch_size, q_len, k_len = attn_mask.size()
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_head, q_len, k_len)
            energy = energy.masked_fill(attn_mask == 0, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x

class RNN(nn.Module):
    def __init__(self, d_model):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
    def forward(self, x, masks, lengths):
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=x.size(1))
        return x

class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled


if __name__ == '__main__':
    m = nn.Conv1d(16, 33, 3, stride=1, padding=1)
    input = torch.randn(20, 16, 50)
    output = m(input, None, None)
    print(output)