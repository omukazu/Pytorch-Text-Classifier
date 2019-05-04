import os
import sys
from typing import Any, Dict, Tuple

import numpy
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss

from data_loader import PNDataLoader
from classifiers import MLP, LSTM, LSTMAttn, CNN, Transformer
from model_components import ScheduledOptimizer


def loss_function(output: torch.Tensor,  # (batch, n_class)
                  target: torch.Tensor,  # (batch)
                  ) -> torch.Tensor:
    softmax = F.softmax(output, dim=-1)  # (batch, n_class)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    """
        (for document classification)
        place_holder = CrossEntropyLoss(reduction='sum')
        loss = place_holder(output, target)
    """
    return loss


def accuracy(output: torch.Tensor,  # (batch, n_class)
             target: torch.Tensor   # (batch)
             ) -> int:
    prediction = torch.argmax(output, dim=1)
    return (prediction == target).sum().item()


def f_measure(output: torch.Tensor,  # (batch, n_class)
              target: torch.Tensor   # (batch)
              ) -> int:
    prediction = torch.argmax(output, dim=1)
    f_score = f1_score(target.cpu(), prediction.cpu(), average='macro')
    return f_score


def load_vocabulary(path: str
                    ) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r") as f:
        word_to_id = {f'{key.strip()}': i + 2 for i, key in enumerate(f)}
        id_to_word = {i + 2: f'{key.strip()}' for i, key in enumerate(f)}
    word_to_id['<PAD>'] = 0
    word_to_id['<UNK>'] = 1
    id_to_word[0] = '<PAD>'
    id_to_word[1] = '<UNK>'
    return word_to_id, id_to_word


def ids_to_embeddings(word_to_id: Dict[str, int],
                      w2v: KeyedVectors
                      ) -> torch.Tensor:
    embeddings = numpy.zeros((len(word_to_id), w2v.vector_size), 'f')  # (vocab_size, d_emb)
    for w, i in word_to_id.items():
        if w == '<PAD>':
            pass  # zero vector
        elif w in w2v.vocab:
            embeddings[i] = w2v.word_vec(w)
        else:
            embeddings[i] = w2v.word_vec('<UNK>')
    return torch.tensor(embeddings)


def load_setting(config: Dict[str, Dict[str, str or int]],
                 args  # argparse.Namespace
                 ):
    torch.manual_seed(config['arguments']['seed'])

    path = 'debug' if args.debug else 'sentences'
    word_to_id, _ = load_vocabulary(config[path]['vocabulary'])
    w2v = KeyedVectors.load_word2vec_format(config[path]['w2v'], binary=True)
    embeddings = ids_to_embeddings(word_to_id, w2v)

    if config['arguments']['model_name'] == 'MLP':
        model = MLP(d_emb=config['arguments']['d_emb'],
                    d_hidden=config['arguments']['d_hidden'],
                    embeddings=embeddings)
    elif config['arguments']['model_name'] == 'LSTM':
        model = LSTM(d_emb=config['arguments']['d_emb'],
                     d_hidden=config['arguments']['d_hidden'],
                     embeddings=embeddings)
    elif config['arguments']['model_name'] == 'LSTMAttn':
        model = LSTMAttn(d_emb=config['arguments']['d_emb'],
                         d_hidden=config['arguments']['d_hidden'],
                         embeddings=embeddings)
    elif config['arguments']['model_name'] == 'CNN':
        model = CNN(d_emb=config['arguments']['d_emb'],
                    embeddings=embeddings,
                    kernel_widths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                    max_seq_len=config['arguments']['max_seq_len'])
    elif config['arguments']['model_name'] == 'Transformer':
        model = Transformer(d_emb=config['arguments']['d_emb'],
                            embeddings=embeddings,
                            max_seq_len=config['arguments']['max_seq_len'])
    else:
        print(f'Unknown model name: {config["arguments"]["model_name"]}', file=sys.stderr)
        return

    # setup device
    if args.gpu and torch.cuda.is_available():
        assert all([int(gpu_number) >= 0 for gpu_number in args.gpu.split(',')]), 'invalid input'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if len(args.gpu) > 1:
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model)
        else:
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # setup data_loader instances
    train_data_loader = PNDataLoader(config[path]['train'], word_to_id, config['arguments']['max_seq_len'],
                                     batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)
    valid_data_loader = PNDataLoader(config[path]['valid'], word_to_id, config['arguments']['max_seq_len'],
                                     batch_size=config['arguments']['batch_size'], shuffle=False, num_workers=2)

    # build optimizer
    if config['arguments']['model_name'] == 'Transformer':
        # filter(lambda x: x.requires_grad, model.parameters()) = extract parameters to be updated
        optimizer = ScheduledOptimizer(torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                                        betas=(0.9, 0.98), eps=1e-09),
                                       config['arguments']['d_emb'],
                                       warmup_steps=4000)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['arguments']['learning_rate'])

    return model, device, train_data_loader, valid_data_loader, optimizer


def load_tester(config: Dict[str, Dict[str, str or int]],
                args  # argparse.Namespace
                ):
    # build model architecture first
    if config['arguments']['model_name'] == 'MLP':
        model = MLP(d_emb=config['arguments']['d_emb'],
                    d_hidden=config['arguments']['d_hidden'],
                    embeddings=None)
    elif config['arguments']['model_name'] == 'BiLSTM':
        model = LSTM(d_emb=config['arguments']['d_emb'],
                     d_hidden=config['arguments']['d_hidden'],
                     embeddings=None)
    elif config['arguments']['model_name'] == 'BiLSTMAttn':
        model = LSTMAttn(d_emb=config['arguments']['d_emb'],
                         d_hidden=config['arguments']['d_hidden'],
                         embeddings=None)
    elif config['arguments']['model_name'] == 'CNN':
        model = CNN(d_emb=config['arguments']['d_emb'],
                    embeddings=None,
                    kernel_widths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                    max_seq_len=config['arguments']['max_seq_len'])
    elif config['arguments']['model_name'] == 'Transformer':
        model = Transformer(d_emb=config['arguments']['d_emb'],
                            embeddings=None,
                            max_seq_len=config['arguments']['max_seq_len'])
    else:
        print(f'Unknown model name: {config["arguments"]["model_name"]}', file=sys.stderr)
        return

    # setup device
    if args.gpu and torch.cuda.is_available():
        assert all([int(gpu_number) >= 0 for gpu_number in args.gpu.split(',')]), 'invalid input'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if len(args.gpu) > 1:
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model, device_ids=args.gpu.split(','))
        else:
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    # load state dict
    state_dict = torch.load(config['arguments']['load_path'], map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)

    # setup data_loader instances
    path = 'debug' if args.debug else 'data'
    word_to_id, _ = load_vocabulary(config[path]['vocabulary'])

    test_data_loader = PNDataLoader(config[path]['test'], word_to_id, config['arguments']['max_seq_len'],
                                    batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)

    # build optimizer
    return model, device, test_data_loader


def create_save_file_name(config: Dict[str, Dict[str, str or int]],
                          params: Dict[str, Any]
                          ) -> str:
    d = config['arguments']
    base = f'{d["model_name"]}-d_hidden:{d["d_hidden"]}-max_seq_len:{d["max_seq_len"]}'
    attributes = "-".join([f'{k}:{v}' for k, v in params.items()])
    return base + '-' + attributes
