import os
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy
from gensim.models import KeyedVectors
import torch

from data_loader import MyDataLoader
from classifiers import SelfAttentionLSTM, CNN, TransformerEncoder
from miscellaneous.constants import UNK
from model_components import ScheduledOptimizer


def load_vocabulary(path: str
                    ) -> Dict[str, int]:
    with open(path, "r") as f:
        word_to_id = {f'{key.strip()}': i + 1 for i, key in enumerate(f)}
    word_to_id['<UNK>'] = UNK
    return word_to_id


def ids_to_embeddings(word_to_id: Dict[str, int],
                      w2v: KeyedVectors
                      ) -> torch.Tensor:
    embeddings = numpy.zeros((len(word_to_id), w2v.vector_size), 'f')  # (vocab_size, d_emb)
    for w, i in word_to_id.items():
        if w in w2v.vocab:
            embeddings[i] = w2v.word_vec(w)
        else:
            embeddings[i] = w2v.word_vec('<UNK>') / i
    return torch.tensor(embeddings)


def load_setting(config: Dict[str, Dict[str, str or int]],
                 args  # argparse.Namespace
                 ) -> Tuple[Any, Any, Any, Any, Any]:
    torch.manual_seed(config["arguments"]["seed"])

    path = "debug" if args.debug else "documents"
    word_to_id = load_vocabulary(config[path]["vocabulary"])
    w2v = KeyedVectors.load_word2vec_format(config[path]["w2v"], binary=True)
    embeddings = ids_to_embeddings(word_to_id, w2v)
    config["arguments"]["vocab_size"] = len(embeddings)

    if config["arguments"]["model_name"] == "CNN":
        model = CNN(d_emb=config["arguments"]["d_emb"],
                    embeddings=embeddings,
                    kernel_widths=[1, 3, 5],
                    n_class=config["arguments"]["n_class"])
    elif config["arguments"]["model_name"] == "LSTM":
        model = SelfAttentionLSTM(d_emb=config["arguments"]["d_emb"],
                                  d_hid=config["arguments"]["d_hid"],
                                  embeddings=embeddings,
                                  n_class=config["arguments"]["n_class"])
    elif config["arguments"]["model_name"] == "Transformer":
        model = TransformerEncoder(d_emb=config["arguments"]["d_emb"],
                                   embeddings=embeddings,
                                   max_seq_len=config["arguments"]["max_seq_len"],
                                   n_class=config["arguments"]["n_class"])
    else:
        raise KeyError(f'Unknown model name: {config["arguments"]["model_name"]}')

    # setup device
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # setup data_loader instances
    train_data_loader = MyDataLoader(config[path]["train"], config[path]["labels"], config["arguments"]["delimiter"],
                                     word_to_id, config["arguments"]["max_seq_len"],
                                     batch_size=config["arguments"]["batch_size"], shuffle=True, num_workers=2)
    valid_data_loader = MyDataLoader(config[path]["valid"], config[path]["labels"], config["arguments"]["delimiter"],
                                     word_to_id, config["arguments"]["max_seq_len"],
                                     batch_size=config["arguments"]["batch_size"], shuffle=False, num_workers=2)

    # build optimizer
    if config["arguments"]["model_name"] == "Transformer":
        # filter(lambda x: x.requires_grad, model.parameters()) = extract parameters to be updated
        optimizer = ScheduledOptimizer(torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                                        betas=(0.9, 0.98), eps=1e-09),
                                       config["arguments"]["d_emb"],
                                       warmup_steps=4000)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["arguments"]["learning_rate"])

    return model, device, train_data_loader, valid_data_loader, optimizer


def load_tester(config: Dict[str, Dict[str, str or int]],
                args  # argparse.Namespace
                ) -> Tuple[Any, Any, Any]:
    # build model architecture first
    if config["arguments"]["model_name"] == "CNN":
        model = CNN(d_emb=config["arguments"]["d_emb"],
                    embeddings=config["arguments"]["vocab_size"],
                    kernel_widths=config["params"]["KernelWidths"],
                    n_class=config["arguments"]["n_class"])
    elif config["arguments"]["model_name"] == "LSTM":
        model = SelfAttentionLSTM(d_emb=config["arguments"]["d_emb"],
                                  d_hid=config["arguments"]["d_hid"],
                                  embeddings=config["arguments"]["vocab_size"],
                                  n_class=config["arguments"]["n_class"])
    elif config["arguments"]["model_name"] == "Transformer":
        model = TransformerEncoder(d_emb=config["arguments"]["d_emb"],
                                   embeddings=config["arguments"]["vocab_size"],
                                   max_seq_len=config["arguments"]["max_seq_len"],
                                   n_class=config["arguments"]["n_class"])
    else:
        raise KeyError(f'Unknown model name: {config["arguments"]["model_name"]}')

    # setup device
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    # load state dict
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)

    # setup data_loader instances
    path = "debug" if args.debug else "documents"
    word_to_id = load_vocabulary(config[path]["vocabulary"])

    test_data_loader = MyDataLoader(config[path]["test"], config[path]["labels"], config["arguments"]["delimiter"],
                                    word_to_id, config["arguments"]["max_seq_len"],
                                    batch_size=config["arguments"]["batch_size"], shuffle=True, num_workers=2)

    # build optimizer
    return model, device, test_data_loader


def create_save_file_name(config: Dict[str, Dict[str, str or int]],
                          params: Dict[str, Any]
                          ) -> str:
    d = config["arguments"]
    base = f'{d["model_name"]}-d_hid:{d["d_hid"]}-max_seq_len:{d["max_seq_len"]}'
    attributes = "-".join([f'{k}:{v}' for k, v in params.items()])
    return base + '-' + attributes


def create_config(config: Dict[str, Dict[str, str or int]],
                  params: Dict[str, Any]
                  ) -> Dict[str, Dict[str, str or int]]:
    save_config = OrderedDict()
    save_config["arguments"] = config["arguments"]
    save_config["documents"] = {"labels": config["documents"]["labels"],
                                "vocabulary": config["documents"]["vocabulary"],
                                "w2v": config["documents"]["w2v"],
                                "test": config["documents"]["test"]}
    save_config["debug"] = {"labels": config["documents"]["labels"],
                            "vocabulary": config["debug"]["vocabulary"],
                            "w2v": config["debug"]["w2v"],
                            "test": config["debug"]["test"]}
    save_config["params"] = params
    return save_config
