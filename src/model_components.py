import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---for RNN---
class RNNWrapper(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings: torch.Tensor or None,
                 rnn: nn.Module,
                 vocab_size: int,
                 dropout_rate: float = 0.333):
        super(RNNWrapper, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings,
                                                  freeze=True) if embeddings is not None \
            else nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, padding_idx=0)
        self.rnn = rnn
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        embedded = self.dropout(self.embed(x))
        lengths = mask.cumsum(dim=1)[:, -1]
        sorted_lengths, perm_indices = lengths.sort(0, descending=True)
        _, unperm_indices = perm_indices.sort(0)

        # masking
        packed = pack_padded_sequence(embedded[perm_indices], lengths=sorted_lengths, batch_first=True)
        output, _ = self.rnn(packed)                                                  # (sum(lengths), hid*2)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)  # (batch, len, hid * 2)
        return self.dropout(unpacked[unperm_indices])                                 # (batch, len, hid * 2)


# ---for CNN---
class CNNComponent(nn.Module):
    def __init__(self,
                 d_emb: int,
                 kernel_width: int,
                 max_seq_len: int,
                 n_filter: int = 128):
        super(CNNComponent, self).__init__()
        self.max_seq_len = max_seq_len
        self.cnn = nn.Conv2d(in_channels=1, out_channels=n_filter, kernel_size=(kernel_width, d_emb))
        self.bn = nn.BatchNorm2d(n_filter, 1)
        self.pool = nn.MaxPool2d(max_seq_len)

    def forward(self,
                x: torch.Tensor,     # (batch, 1, max_seq_len, d_emb)
                mask: torch.Tensor,  # (batch, max_seq_len)
                ) -> torch.Tensor:
        cnn = self.cnn(x)                   # (batch, num_filters, max_seq_len)
        bn = F.relu(self.bn(cnn))           # (batch, num_filters, max_seq_len)
        pooled = self.pool(bn).squeeze(-1)  # (batch, num_filters)
        return pooled


# ---for Transformer---
class ScheduledOptimizer:
    def __init__(self,
                 optimizer: torch.optim,
                 d_emb: int,
                 warmup_steps: int):
        self._optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.initial_lr = np.power(d_emb, -0.5)

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.current_step += 1
        scale = np.min([np.power(self.current_step, -0.5), np.power(self.warmup_steps, -1.5) * self.current_step])
        lr = self.initial_lr * scale

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Embedder(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings: torch.Tensor or None,
                 max_seq_len: int,
                 vocab_size: int):
        super(Embedder, self).__init__()
        self.d_emb = d_emb
        self.max_seq_len = max_seq_len
        self.embeddings = nn.Embedding.from_pretrained(embeddings,
                                                       freeze=True) if embeddings is not None \
            else nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, padding_idx=0)
        self.positional_encoding = nn.Embedding.from_pretrained(self.create_pe_embeddings(self.d_emb, max_seq_len))

    def forward(self,
                x: torch.Tensor,    # (batch, max_seq_len)
                mask: torch.Tensor  # (batch, max_seq_len)
                ) -> torch.Tensor:
        # (batch, max_seq_len), mask -> position e.g. (1,1,1,0,0) -> (1,2,3,0,0)
        position = mask.cumsum(dim=1) * mask
        return self.embeddings(x) + self.positional_encoding(position)

    @staticmethod
    def create_pe_embeddings(d_emb: int,
                             max_seq_len: int,
                             padding_index: int = 0
                             ) -> torch.Tensor:
        # (max_seq_len, d_emb), position -> embedding_vector
        pe_embeddings = torch.tensor([[(position - 1) / np.power(10000, 2 * i_emb / d_emb) for i_emb in range(d_emb)]
                                      for position in range(max_seq_len + 1)])
        pe_embeddings[1:, 0::2] = torch.sin(pe_embeddings[1:, 0::2])
        pe_embeddings[1:, 1::2] = torch.cos(pe_embeddings[1:, 1::2])
        pe_embeddings[padding_index] = 0
        return pe_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hidden: int,
                 dropout_rate: float,
                 n_head: int):
        super(MultiHeadAttention, self).__init__()
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(p=dropout_rate)
        self.n_head = n_head

        self.w_q = nn.Linear(d_emb, n_head * d_hidden, bias=False)
        self.w_k = nn.Linear(d_emb, n_head * d_hidden, bias=False)
        self.w_v = nn.Linear(d_emb, n_head * d_hidden, bias=False)
        self.w_o = nn.Linear(n_head * d_hidden, d_emb, bias=False)
        self.normalize = nn.LayerNorm(d_emb, eps=1e-6)

    def forward(self,
                x: torch.Tensor,    # (batch, max_seq_len, d_emb)
                mask: torch.Tensor  # (batch, len)
                ) -> torch.Tensor:
        batch, max_seq_len, _ = x.size()
        n_head, d_hidden = self.n_head, self.d_hidden

        # (batch, n_head, max_seq_len, d_hidden)
        query = self.w_q(x).contiguous().view(batch, max_seq_len, n_head, d_hidden).transpose(1, 2)
        key = self.w_k(x).contiguous().view(batch, max_seq_len, n_head, d_hidden).transpose(1, 2)
        value = self.w_v(x).contiguous().view(batch, max_seq_len, n_head, d_hidden).transpose(1, 2)
        z = self.scaled_dot_product_attention(query, key, value, mask,
                                              scaling_factor=1 / np.power(d_hidden, 2), n_head=n_head)
        z = z.transpose(1, 2).contiguous().view(batch, max_seq_len, -1)  # (batch, max_seq_len, n_head * d_hidden)
        z = self.dropout(self.w_o(z))                                    # (batch, max_seq_len, d_emb)
        return self.normalize(x + z)                                     # residual and normalization

    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor,  # (batch, n_head, max_seq_len, d_hidden)
                                     key: torch.Tensor,    # (batch, n_head, max_seq_len, d_hidden)
                                     value: torch.Tensor,  # (batch, n_head, max_seq_len, d_hidden)
                                     mask: torch.Tensor,   # (batch, max_seq_len)
                                     scaling_factor: np.float64,
                                     n_head: int
                                     ) -> torch.Tensor:
        # (batch, n_head, max_seq_len, max_seq_len)
        alignment_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling_factor

        if mask is not None:
            tensor_type = 'torch.cuda.FloatTensor' if alignment_weights.device.index >= 0 else 'torch.FloatTensor'
            mask = mask.unsqueeze(2).type(tensor_type)         # (batch, max_seq_len, 1)
            _mask = torch.bmm(mask, mask.transpose(1, 2))            # (batch, max_seq_len, max_seq_len)
            _mask = _mask.unsqueeze(1).expand((-1, n_head, -1, -1))  # (batch, n_head, max_seq_len, max_seq_len)
            alignment_weights.masked_fill_(_mask.ne(1), -1e6)

        alignment_weights = F.softmax(alignment_weights, dim=-1)
        self_attention = torch.matmul(alignment_weights, value)      # (batch, n_head, max_seq_len, d_hidden)
        return self_attention


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hidden: int,
                 dropout_rate: float):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.w_1 = nn.Conv1d(d_emb, d_hidden, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.w_2 = nn.Conv1d(d_hidden, d_emb, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.normalize = nn.LayerNorm(d_emb, eps=1e-6)

    def forward(self,
                x: torch.Tensor,  # (batch, max_seq_len, d_emb)
                ) -> torch.Tensor:
        h = x.transpose(1, 2)                   # (batch, d_emb, max_seq_len)
        h = self.dropout1(F.relu(self.w_1(h)))  # (batch, d_hidden, max_seq_len)
        h = self.dropout2(self.w_2(h))          # (batch, d_emb, max_seq_len)
        h = h.transpose(1, 2)                   # (batch, max_seq_len, d_emb)
        return self.normalize(x + h)            # residual and normalization


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_emb: int,
                 dropout_rate: float,
                 n_head: int = 8,
                 scale: int = 4):
        super(EncoderLayer, self).__init__()
        self.n_head = n_head
        self.scale = scale
        assert d_emb % self.n_head == 0, 'invalid d_emb'

        self.multi_head_attention = MultiHeadAttention(d_emb, d_emb // self.n_head,
                                                       n_head=n_head, dropout_rate=dropout_rate)
        self.position_wise_feed_forward_network = PositionWiseFeedForwardNetwork(d_emb, d_emb * self.scale,
                                                                                 dropout_rate=dropout_rate)

    def forward(self,
                x: torch.Tensor,    # (batch, max_seq_len, d_emb)
                mask: torch.Tensor  # (batch, max_seq_len)
                ) -> torch.Tensor:
        self_attention = self.multi_head_attention(x, mask)               # (batch, max_seq_len, d_emb)
        output = self.position_wise_feed_forward_network(self_attention)  # (batch, max_seq_len, d_emb)
        return output
