import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

from data_loader import MyDataLoader
from classifiers import MLP, LSTM, SelfAttentionLSTM, CNN, TransformerEncoder
from constants import UNK
from model_components import ScheduledOptimizer


def loss_function(output: torch.Tensor,  # (b, n_class)
                  target: torch.Tensor,  # (b)
                  ) -> torch.Tensor:
    softmax = F.softmax(output, dim=-1)  # (b, n_class)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    """
        # for document classification
        loss = F.crossentropy(output, target, reduction='none').sum() / b
    """
    return loss


def accuracy(output: torch.Tensor,  # (b, n_class)
             target: torch.Tensor   # (b)
             ) -> int:
    prediction = torch.argmax(output, dim=1)
    return (prediction == target).sum().item()


def f_measure(output: torch.Tensor,  # (b, n_class)
              target: torch.Tensor   # (b)
              ) -> int:
    prediction = torch.argmax(output, dim=1)
    f_score = f1_score(target.cpu(), prediction.cpu(), average='macro')
    return f_score


def load_vocabulary(path: str
                    ) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r") as f:
        word_to_id = {f'{key.strip()}': i + 1 for i, key in enumerate(f)}
        id_to_word = {i + 1: f'{key.strip()}' for i, key in enumerate(f)}
    word_to_id['<UNK>'] = UNK
    id_to_word[UNK] = '<UNK>'
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
                    d_hid=config['arguments']['d_hid'],
                    embeddings=embeddings)
    elif config['arguments']['model_name'] == 'CNN':
        model = CNN(d_emb=config['arguments']['d_emb'],
                    embeddings=embeddings,
                    kernel_widths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
    elif config['arguments']['model_name'] == 'LSTM':
        model = LSTM(d_emb=config['arguments']['d_emb'],
                     d_hid=config['arguments']['d_hid'],
                     embeddings=embeddings)
    elif config['arguments']['model_name'] == 'SALSTM':
        model = SelfAttentionLSTM(d_emb=config['arguments']['d_emb'],
                                  d_hid=config['arguments']['d_hid'],
                                  embeddings=embeddings)
    elif config['arguments']['model_name'] == 'Transformer':
        model = TransformerEncoder(d_emb=config['arguments']['d_emb'],
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
            ids = list(map(int, args.gpu.split(',')))
            device = torch.device(f'cuda')
            model = torch.nn.DataParallel(model, device_ids=ids)
        else:
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # setup data_loader instances
    train_data_loader = MyDataLoader(config[path]['train'], word_to_id, config['arguments']['max_seq_len'],
                                     batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)
    valid_data_loader = MyDataLoader(config[path]['valid'], word_to_id, config['arguments']['max_seq_len'],
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
                    d_hid=config['arguments']['d_hid'],
                    embeddings=config['arguments']['vocab_size'])
    elif config['arguments']['model_name'] == 'CNN':
        model = CNN(d_emb=config['arguments']['d_emb'],
                    embeddings=config['arguments']['vocab_size'],
                    kernel_widths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20])
    elif config['arguments']['model_name'] == 'LSTM':
        model = LSTM(d_emb=config['arguments']['d_emb'],
                     d_hid=config['arguments']['d_hid'],
                     embeddings=config['arguments']['vocab_size'])
    elif config['arguments']['model_name'] == 'SALSTM':
        model = SelfAttentionLSTM(d_emb=config['arguments']['d_emb'],
                                  d_hid=config['arguments']['d_hid'],
                                  embeddings=config['arguments']['vocab_size'])
    elif config['arguments']['model_name'] == 'Transformer':
        model = TransformerEncoder(d_emb=config['arguments']['d_emb'],
                                   embeddings=config['arguments']['vocab_size'],
                                   max_seq_len=config['arguments']['max_seq_len'])
    else:
        print(f'Unknown model name: {config["arguments"]["model_name"]}', file=sys.stderr)
        return

    # setup device
    if args.gpu and torch.cuda.is_available():
        assert all([int(gpu_number) >= 0 for gpu_number in args.gpu.split(',')]), 'invalid input'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if len(args.gpu) > 1:
            ids = list(map(int, args.gpu.split(',')))
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model, device_ids=ids)
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

    test_data_loader = MyDataLoader(config[path]['test'], word_to_id, config['arguments']['max_seq_len'],
                                    batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)

    # build optimizer
    return model, device, test_data_loader


def create_save_file_name(config: Dict[str, Dict[str, str or int]],
                          params: Dict[str, Any]
                          ) -> str:
    d = config['arguments']
    base = f'{d["model_name"]}-d_hid:{d["d_hid"]}-max_seq_len:{d["max_seq_len"]}'
    attributes = "-".join([f'{k}:{v}' for k, v in params.items()])
    return base + '-' + attributes


def create_config(config: Dict[str, Dict[str, str or int]],
                  params: Dict[str, Any]
                  ) -> Dict[str, Dict[str, str or int]]:
    save_config = OrderedDict()
    save_config['arguments'] = config['arguments']
    save_config['test'] = {"vocabulary": "/mnt/larch_f//vocab.txt",
                           "w2v": "/mnt/windroot/share/word2vec/2016.08.02/w2v.midasi.256.100K.bin",
                           "test": "/mnt/hinoki_f///test.txt"}
    save_config['debug'] = {"vocabulary": "debug/vocab.txt",
                            "w2v": "debug/w2v.midasi.256.100K.bin",
                            "test": "debug/test.txt"}
    save_config['params'] = params
    return save_config
