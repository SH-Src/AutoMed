import torch
import torch.nn.functional as F
from models_new.operations import *
from torch.autograd import Variable
from models_new.genotypes import PRIMITIVES
from models_new.genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

    def __init__(self, d_model):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](d_model)
            self._ops.append(op)

    def forward(self, x, masks, lengths, weights):
        return sum(w * op(x, masks, lengths) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, d_model, steps):
        super(Cell, self).__init__()
        self._steps = steps
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + 1):
                op = MixedOp(d_model)
                self._ops.append(op)

    def forward(self, s0, masks, lengths, weights):
        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, masks, lengths, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states


class Network(nn.Module):
    def __init__(self, vocab_size, d_model, steps, criterion, num_classes=2):
        super(Network, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.embbedding.weight.data.uniform_(-0.04, 0.04)
        self._criterion = criterion
        self._steps = steps
        self._vocab_size = vocab_size
        self.d_model = d_model
        self.time_cell = Cell(d_model, steps)
        self.ehr_cell = Cell(d_model, steps)
        self.fuse_cell = Cell(d_model, steps)
        self.catfc = nn.Linear(2 * d_model, d_model)
        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(0.1)
        self.time_layer = torch.nn.Linear(64, d_model)
        self.selection_layer = torch.nn.Linear(1, 64)
        # self.pooler = MaxPoolLayer()
        self.combine = nn.Linear(steps + 1, 1, bias=False)
        self.weight_layer = nn.Linear(d_model, 1)
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_time = Variable(1e-3 * torch.ones(k, num_ops).cuda(), requires_grad=True)
        self.alphas_ehr = Variable(1e-3 * torch.ones(k, num_ops).cuda(), requires_grad=True)
        self.alphas_fuse = Variable(1e-3 * torch.ones(k, num_ops).cuda(), requires_grad=True)
        self.alphas_select = Variable(1e-3 * torch.ones(2, self._steps + 1).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_time,
            self.alphas_ehr,
            self.alphas_fuse,
            self.alphas_select
        ]

    def new(self):
        model_new = Network(self._vocab_size, self.d_model, self._steps, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, input_seqs, masks, lengths, seq_time_step):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        ehr_states = self.ehr_cell(x, masks, lengths, torch.softmax(self.alphas_ehr, dim=-1))
        time_states = self.time_cell(time_feature, masks, lengths, torch.softmax(self.alphas_time, dim=-1))
        alphas_select = torch.softmax(self.alphas_select, dim=-1)
        ehr_feature = sum(w * s for w, s in zip(alphas_select[0], ehr_states))
        time_feature = sum(w * s for w, s in zip(alphas_select[1], time_states))
        cat_feature = torch.cat((ehr_feature, time_feature), dim=-1)
        fused_feature = self.catfc(cat_feature)
        alpha_fuse = torch.softmax(self.alphas_fuse, dim=-1)
        # alpha_fuse[-self._steps].data.copy_(F.one_hot(torch.tensor(4), num_classes=6).data)
        final_states = self.fuse_cell(fused_feature, masks, lengths, alpha_fuse)
        final_states = self.combine(torch.stack(final_states, dim=-1)).squeeze()
        rnn_input = pack_padded_sequence(final_states, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_input)
        final_states, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        weight = self.weight_layer(final_states)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1)
        weighted_features = final_states * att.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        # output = self.pooler(final_states, lengths)
        output = self.classifier(averaged_features)
        return output

    def _loss(self, input, target):
        logits = self(input[0], input[1], input[2], input[3])
        return self._criterion(logits, target)

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end]
                for j in range(len(W)):
                    k_best = torch.argmax(W[j])
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_time = _parse(torch.softmax(self.alphas_time, dim=-1))
        gene_ehr = _parse(torch.softmax(self.alphas_ehr, dim=-1))
        alpha_fuse = torch.softmax(self.alphas_fuse, dim=-1)
        gene_fuse = _parse(alpha_fuse)
        gene_select = torch.argmax(torch.softmax(self.alphas_select, dim=-1), dim=-1)
        gene_select = [gene_select[0].item(), gene_select[1].item()]

        genotype = Genotype(
            time=gene_time,
            ehr=gene_ehr,
            fuse=gene_fuse,
            select=gene_select
        )
        return genotype


if __name__ == '__main__':
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    model = Network(10, 8, 2, loss_func)
    print(model.genotype())
