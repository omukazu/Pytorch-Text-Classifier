from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import RNNWrapper, CNNComponent, Embedder, EncoderLayer


class MLP(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hidden: int,
                 embeddings: torch.Tensor or None,
                 n_class: int = 2):
        super(MLP, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings,
                                                  freeze=False) if embeddings is not None \
            else nn.Embedding(num_embeddings=50002, embedding_dim=d_emb, padding_idx=0)
        self.w_1 = nn.Linear(d_emb, d_hidden)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hidden, n_class)

        self.params = {}

    def forward(self,
                x: torch.Tensor,    # (batch, max_seq_len, d_emb)
                mask: torch.Tensor  # (batch, max_seq_len)
                ) -> torch.Tensor:  # (batch, n_class)
        embedded = self.embed(x)
        # (batch, d_emb) / (batch, 1) -> (batch, d_emb)
        x_phrase = torch.sum(embedded, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()
        h = self.tanh(self.w_1(x_phrase))  # (b, hid)
        y = self.w_2(h)                    # (batch, n_class)
        return y


class LSTM(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hidden: int,
                 embeddings: torch.Tensor or None,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_layer: int = 1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=d_emb, hidden_size=d_hidden, num_layers=n_layer,
                           batch_first=True, dropout=dropout_rate, bidirectional=bi_directional)
        self.rnn_wrapper = RNNWrapper(d_emb, embeddings, self.rnn)

        self.w_1 = nn.Linear(d_hidden * 2, d_hidden)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hidden, n_class)

        self.params = {'BiDirectional': bi_directional,
                       'DropoutRate': dropout_rate,
                       'NLayer': n_layer}

    def forward(self,
                x: torch.Tensor,     # (batch, max_seq_len, d_emb)
                mask: torch.Tensor,  # (batch, max_seq_len)
                ) -> torch.Tensor:   # (batch, n_class)
        rnn_out = self.rnn_wrapper(x, mask)  # (batch, max_seq_len, d_hidden * 2)
        out = rnn_out.sum(dim=1)             # (batch, d_hidden * 2)
        h = self.tanh(self.w_1(out))         # (batch, d_hidden)
        y = self.w_2(F.dropout(h, p=0.25))   # (batch, n_class)
        return y


class LSTMAttn(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hidden: int,
                 embeddings,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_layer: int = 1):
        super(LSTMAttn, self).__init__()
        self.rnn = nn.LSTM(input_size=d_emb, hidden_size=d_hidden, num_layers=n_layer,
                           batch_first=True, dropout=dropout_rate, bidirectional=bi_directional)
        self.rnn_wrapper = RNNWrapper(d_emb, embeddings, self.rnn)

        self.attention = nn.Linear(d_hidden * 2, 1)
        self.w_1 = nn.Linear(d_hidden * 2, d_hidden)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hidden, n_class)

        self.params = {'BiDirectional': bi_directional,
                       'DropoutRate': dropout_rate,
                       'NLayer': n_layer}

    def forward(self,
                x: torch.Tensor,     # (batch, max_seq_len, d_emb)
                mask: torch.Tensor,  # (batch, max_seq_len)
                ) -> torch.Tensor:   # (batch, n_class)
        rnn_out = self.rnn_wrapper(x, mask)                                  # (batch, max_seq_len, d_hidden * 2)
        # attn = F.softmax(self.l_attn(rnn_out), dim=1)
        alignment_weights = self.calculate_alignment_weights(rnn_out, mask)  # (batch, max_seq_len, 1)
        out = (alignment_weights * rnn_out).sum(dim=1)                       # (batch, d_hidden * 2)
        h = self.tanh(self.w_1(out))                                         # (batch, d_hidden)
        y = self.w_2(h)                                                      # (batch, n_class)
        return y

    def calculate_alignment_weights(self,
                                    rnn_out: torch.Tensor,  # (batch, max_seq_len, d_hidden * 2)
                                    mask: torch.Tensor      # (batch, max_seq_len)
                                    ) -> torch.Tensor:
        max_len = rnn_out.size(1)
        alignment_weights = self.attention(rnn_out)    # (batch, max_seq_len, 1)
        alignment_weights_mask = mask.unsqueeze(-1).type(torch.FloatTensor)
        index = alignment_weights.device.index
        if index:  # to gpu
            alignment_weights_mask = alignment_weights_mask.to(torch.device(f'cuda:{index}'))
        alignment_weights.masked_fill_(alignment_weights_mask[:, :max_len, :].ne(1), -1e12)
        return F.softmax(alignment_weights, dim=1)  # (b, len, 1)


class CNN(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings,
                 kernel_widths: List[int],
                 max_seq_len: int,
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_filter: int = 128):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings,
                                                  freeze=False) if embeddings is not None \
            else nn.Embedding(num_embeddings=50002, embedding_dim=d_emb, padding_idx=0)
        assert len(kernel_widths) > 1, 'kernel_widths need at least two elements'
        n_kernel = len(kernel_widths)
        self.poolings = nn.ModuleList([CNNComponent(d_emb=d_emb,
                                                    kernel_width=kernel_widths[i],
                                                    max_seq_len=max_seq_len)
                                      for i in range(n_kernel)])

        # highway architecture
        self.sigmoid = nn.Sigmoid()
        self.transform_gate = nn.Linear(n_filter * n_kernel, n_filter * n_kernel)
        self.highway = nn.Linear(n_filter * n_kernel, n_filter * n_kernel)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(n_filter * n_kernel, n_class)

        self.params = {'DropoutRate': dropout_rate,
                       'KernelWidths': ','.join(map(str, kernel_widths)),
                       'NFilter': n_filter}

    def forward(self,
                x: torch.Tensor,     # (batch, len, d_emb)
                mask: torch.Tensor,  # (batch, len)
                ) -> torch.Tensor:
        embedded = self.embed(x)
        embedded = embedded.unsqueeze(1)  # (batch, 1, max_seq_len, d_emb)
        pooled = self.poolings[0](embedded, mask)
        for pooling in self.poolings[1:]:
            pooled = torch.cat((pooled, pooling(embedded, mask)), dim=1)
        pooled = pooled.squeeze(-1)       # (batch, num_filters * n_kernel)

        t = self.sigmoid(self.transform_gate(pooled))             # (batch, num_filters * n_kernel)
        hw = t * F.relu(self.highway(pooled)) + (1 - t) * pooled  # (batch, num_filters * n_kernel)

        h = self.fc(self.dropout(hw))                             # (batch, n_class)
        return h


class Transformer(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings,
                 dropout_rate: float = 0.333,
                 max_seq_len: int = None,
                 n_class: int = 2,
                 n_layer: int = 6):
        super(Transformer, self).__init__()
        self.embedder = Embedder(d_emb, embeddings, max_seq_len)
        self.encoder_layer = nn.ModuleList([EncoderLayer(d_emb, dropout_rate=dropout_rate) for _ in range(n_layer)])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(d_emb, n_class)

        self.params = {'DropoutRate': dropout_rate,
                       'NHead': self.encoder_layer[0].n_head,
                       'NLayer': n_layer,
                       'Scale': self.encoder_layer[0].scale}

    def forward(self,
                x: List[np.array],
                mask: torch.Tensor
                ) -> torch.Tensor:
        h = self.embedder(x, mask)      # (batch, max_seq_len, d_emb)
        for encoder_layer in self.encoder_layer:
            h = encoder_layer(h, mask)  # (batch, max_seq_len, d_emb)
        h = h[:, -1, :]                 # extract final feature
        y = self.fc(self.dropout(h))    # (batch, n_class)
        return y
