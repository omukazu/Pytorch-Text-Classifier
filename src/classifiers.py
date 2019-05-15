from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import Embedder, CNNPooler, RNNWrapper, TransformerEmbedder, EncoderLayer


class MLP(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hid: int,
                 embeddings: torch.Tensor or int,
                 n_class: int = 2
                 ) -> None:
        super(MLP, self).__init__()
        self.vocab_size = embeddings if type(embeddings) is int else embeddings.size(0)
        self.embed = Embedder(self.vocab_size, d_emb)
        self.embed.set_initial_embedding(embeddings, freeze=True)

        self.w_1 = nn.Linear(d_emb, d_hid)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hid, n_class)

        self.params = {'VocabSize': self.vocab_size}

    def forward(self,
                x: torch.Tensor,    # (b, max_seq_len, d_emb)
                mask: torch.Tensor  # (b, max_seq_len)
                ) -> torch.Tensor:  # (b, n_class)
        embedded = self.embed(x, mask)
        # (b, d_emb) / (b, 1) -> (b, d_emb)
        x_phrase = torch.sum(embedded, dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1).float()
        h = self.tanh(self.w_1(x_phrase))  # (b, d_hid)
        y = self.w_2(h)                    # (b, n_class)
        return y


class CNN(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings: torch.Tensor or int,
                 kernel_widths: List[int],
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_filter: int = 128
                 ) -> None:
        super(CNN, self).__init__()
        self.vocab_size = embeddings if type(embeddings) is int else embeddings.size(0)
        self.embed = Embedder(self.vocab_size, d_emb)
        self.embed.set_initial_embedding(embeddings, freeze=False)
        assert len(kernel_widths) > 1, 'kernel_widths need at least two elements'
        n_kernel = len(kernel_widths)
        self.poolers = nn.ModuleList([CNNPooler(d_emb=d_emb,
                                                kernel_width=kernel_widths[i])
                                      for i in range(n_kernel)])

        # highway architecture
        self.sigmoid = nn.Sigmoid()
        self.transform_gate = nn.Linear(n_filter * n_kernel, n_filter * n_kernel)
        self.highway = nn.Linear(n_filter * n_kernel, n_filter * n_kernel)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(n_filter * n_kernel, n_class)

        self.params = {'DropoutRate': dropout_rate,
                       'KernelWidths': ','.join(map(str, kernel_widths)),
                       'NFilter': n_filter,
                       'VocabSize': self.vocab_size}

    def forward(self,
                x: torch.Tensor,     # (b, len, d_emb)
                mask: torch.Tensor,  # (b, len)
                ) -> torch.Tensor:
        embedded = self.embed(x, mask)
        embedded = embedded.unsqueeze(1)  # (b, 1, max_seq_len, d_emb)
        pooled = self.poolers[0](embedded, mask)
        for pooler in self.poolers[1:]:
            pooled = torch.cat((pooled, pooler(embedded, mask)), dim=1)
        pooled = pooled.squeeze(-1)       # (b, num_filters * n_kernel)

        t = self.sigmoid(self.transform_gate(pooled))             # (b, num_filters * n_kernel)
        hw = t * F.relu(self.highway(pooled)) + (1 - t) * pooled  # (b, num_filters * n_kernel)

        y = self.fc(self.dropout(hw))                             # (b, n_class)
        return y


class LSTM(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hid: int,
                 embeddings: torch.Tensor or int,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_layer: int = 1
                 ) -> None:
        super(LSTM, self).__init__()
        self.vocab_size = embeddings if type(embeddings) is int else embeddings.size(0)
        self.embed = Embedder(self.vocab_size, d_emb)
        self.embed.set_initial_embedding(embeddings, freeze=True)
        self.rnn = RNNWrapper(nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=n_layer,
                                      batch_first=True, dropout=dropout_rate, bidirectional=bi_directional))

        self.w_1 = nn.Linear(d_hid * 2, d_hid)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hid, n_class)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.params = {'BiDirectional': bi_directional,
                       'DropoutRate': dropout_rate,
                       'NLayer': n_layer,
                       'VocabSize': self.vocab_size}

    def forward(self,
                x: torch.Tensor,     # (b, max_seq_len, d_emb)
                mask: torch.Tensor,  # (b, max_seq_len)
                ) -> torch.Tensor:   # (b, n_class)
        embedded = self.embed(x, mask)
        rnn_out = self.rnn(embedded, mask)  # (b, max_seq_len, d_hid * 2)
        out = rnn_out.sum(dim=1)            # (b, d_hid * 2)
        h = self.tanh(self.w_1(out))        # (b, d_hid)
        y = self.w_2(self.dropout(h))       # (b, n_class)
        return y


class SelfAttentionLSTM(nn.Module):
    def __init__(self,
                 d_emb: int,
                 d_hid: int,
                 embeddings: torch.Tensor or int,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_class: int = 2,
                 n_layer: int = 1
                 ) -> None:
        super(SelfAttentionLSTM, self).__init__()
        self.vocab_size = embeddings if type(embeddings) is int else embeddings.size(0)
        self.embed = Embedder(self.vocab_size, d_emb)
        self.embed.set_initial_embedding(embeddings, freeze=True)
        self.rnn = RNNWrapper(nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=n_layer,
                                      batch_first=True, dropout=dropout_rate, bidirectional=bi_directional))

        self.attention = nn.Linear(d_hid * 2, 1)
        self.w_1 = nn.Linear(d_hid * 2, d_hid)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_hid, n_class)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.params = {'BiDirectional': bi_directional,
                       'DropoutRate': dropout_rate,
                       'NLayer': n_layer,
                       'VocabSize': self.vocab_size}

    def forward(self,
                x: torch.Tensor,     # (b, max_seq_len, d_emb)
                mask: torch.Tensor,  # (b, max_seq_len)
                ) -> torch.Tensor:   # (b, n_class)
        embedded = self.embed(x, mask)
        rnn_out = self.rnn(embedded, mask)                                   # (b, seq_len, d_hid * 2)
        # alignment_weights = F.softmax(self.attention(rnn_out), dim=1)
        alignment_weights = self.calculate_alignment_weights(rnn_out, mask)  # (b, seq_len, 1)
        out = (alignment_weights * rnn_out).sum(dim=1)                       # (b, d_hid * 2)
        h = self.tanh(self.w_1(out))                                         # (b, d_hid)
        y = self.w_2(self.dropout(h))                                        # (b, n_class)
        return y

    def calculate_alignment_weights(self,
                                    rnn_out: torch.Tensor,  # (b, max_seq_len, d_hid * 2)
                                    mask: torch.Tensor      # (b, max_seq_len)
                                    ) -> torch.Tensor:
        max_len = rnn_out.size(1)
        alignment_weights = self.attention(rnn_out)  # (b, seq_len, 1)
        alignment_weights_mask = mask.unsqueeze(-1).type(alignment_weights.dtype)
        alignment_weights.masked_fill_(alignment_weights_mask[:, :max_len, :].ne(1), -1e6)
        return F.softmax(alignment_weights, dim=1)   # (b, seq_len, 1)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings: torch.Tensor or int,
                 dropout_rate: float = 0.333,
                 max_seq_len: int = None,
                 n_class: int = 2,
                 n_layer: int = 6
                 ) -> None:
        super(TransformerEncoder, self).__init__()
        self.vocab_size = embeddings if type(embeddings) is int else embeddings.size(0)
        self.embed = TransformerEmbedder(self.vocab_size, d_emb, max_seq_len)
        self.embed.set_initial_embedding(embeddings, freeze=True)
        self.encoder_layer = nn.ModuleList([EncoderLayer(d_emb, dropout_rate=dropout_rate) for _ in range(n_layer)])

        self.self_attention = nn.Linear(d_emb, 1)
        self.w_1 = nn.Linear(d_emb, d_emb)
        self.tanh = nn.Tanh()
        self.w_2 = nn.Linear(d_emb, n_class)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.params = {'DropoutRate': dropout_rate,
                       'NHead': self.encoder_layer[0].n_head,
                       'NLayer': n_layer,
                       'Scale': self.encoder_layer[0].scale,
                       'VocabSize': self.vocab_size}

    def forward(self,
                x: List[np.array],  # (b, max_seq_len, d_emb)
                mask: torch.Tensor  # (b, max_seq_len)
                ) -> torch.Tensor:
        h = self.embed(x, mask)                                        # (b, max_seq_len, d_emb)
        for encoder_layer in self.encoder_layer:
            h = encoder_layer(h, mask)                                 # (b, max_seq_len, d_emb)
        alignment_weights = self.calculate_alignment_weights(h, mask)  # (b, max_seq_len, 1)
        h = (alignment_weights * h).sum(dim=1)                         # (b, d_hid)
        h = self.tanh(self.w_1(h))                                     # (b, d_hid)
        y = self.w_2(self.dropout(h))                                  # (b, n_class)
        return y

    def calculate_alignment_weights(self,
                                    x: torch.Tensor,        # (b, max_seq_len, d_emb)
                                    mask: torch.Tensor      # (b, max_seq_len)
                                    ) -> torch.Tensor:
        max_len = x.size(1)
        alignment_weights = self.self_attention(x)  # (b, max_seq_len, 1)
        alignment_weights_mask = mask.unsqueeze(-1).type(alignment_weights.dtype)
        alignment_weights.masked_fill_(alignment_weights_mask[:, :max_len, :].ne(1), -1e6)
        return F.softmax(alignment_weights, dim=1)  # (b, max_seq_len, 1)
