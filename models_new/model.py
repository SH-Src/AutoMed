import torch
import torch.nn as nn
from models_new.operations import *


class Cell(nn.Module):

    def __init__(self, genotype, d_model, step):
        super(Cell, self).__init__()
        op_names, indices = zip(*genotype)
        self._compile(op_names, indices, step, d_model)

    def _compile(self, op_names, indices, step, d_model):
        assert len(op_names) == len(indices)
        self._steps = step
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](d_model)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, masks, lengths):

        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, masks, lengths) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states


class Network(nn.Module):
    def __init__(self, vocab_size, d_model, steps, criterion, genotype, num_classes=2):
        super(Network, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.embbedding.weight.data.uniform_(-0.04, 0.04)
        self._criterion = criterion
        self._steps = steps
        self._vocab_size = vocab_size
        self.d_model = d_model
        self.time_cell = Cell(genotype.time, d_model, steps)
        self.ehr_cell = Cell(genotype.ehr, d_model, steps)
        self.fuse_cell = Cell(genotype.fuse, d_model, steps)
        self.catfc = nn.Linear(2 * d_model, d_model)
        self.select = genotype.select
        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(0.1)
        self.time_layer = torch.nn.Linear(64, d_model)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.pooler = MaxPoolLayer()
        self.combine = nn.Linear(steps + 1, 1, bias=False)
        self.weight_layer = nn.Linear(d_model, 1)
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_seqs, masks, lengths, seq_time_step):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        ehr_states = self.ehr_cell(x, masks, lengths)
        time_states = self.time_cell(time_feature, masks, lengths)
        cat_feature = torch.cat((ehr_states[self.select[0]], time_states[self.select[1]]), dim=-1)
        fused_feature = self.catfc(cat_feature)
        # fused_feature = ehr_states[self.select[0]] + time_states[self.select[1]]
        final_states = self.fuse_cell(fused_feature, masks, lengths)
        final_states = self.combine(torch.stack(final_states, dim=-1)).squeeze()
        rnn_input = pack_padded_sequence(final_states, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_input)
        final_states, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        weight = self.weight_layer(final_states)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1)
        weighted_features = final_states * att.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        # output = self.pooler(final_states[-1], lengths)
        output = self.classifier(averaged_features)
        return output