import json
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import torch

from miscellaneous.metrics import loss_function, accuracy, f_measure
from miscellaneous.utils import load_tester


def main():
    # Testing setting
    parser = ArgumentParser(description='test a classifier', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--model', default=None, type=str, help='path to model')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers\nto specify')
    parser.add_argument('--debug', '-d', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    model, device, test_data_loader = load_tester(config, args)

    # test
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_f_score = 0
        num_iter = 0
        for batch_idx, (source, mask, target) in enumerate(test_data_loader):
            source = source.to(device)
            mask = mask.to(device)
            target = target.to(device)

            output = model(source, mask)

            total_loss += loss_function(output, target)
            total_accuracy += accuracy(output, target)
            total_f_score += f_measure(output, target)
            num_iter = batch_idx + 1
    print(f'test_loss={total_loss / test_data_loader.n_samples:.3f}', end=' ')
    print(f'test_accuracy={total_accuracy / test_data_loader.n_samples:.3f}', end=' ')
    print(f'test_f_score={total_f_score / num_iter:.3f}\n')


if __name__ == '__main__':
    main()
