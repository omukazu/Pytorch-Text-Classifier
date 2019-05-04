import json
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from utils import loss_function, accuracy, f_measure, load_setting, create_save_file_name


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    os.makedirs(os.path.dirname(config['arguments']['save_path']), exist_ok=True)

    model, device, train_data_loader, valid_data_loader, optimizer = load_setting(config, args)
    file_name = create_save_file_name(config, model.params)

    best_acc = 0

    for epoch in range(1, config['arguments']['epoch'] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (source, mask, target) in tqdm(enumerate(train_data_loader)):
            source = source.to(device)  # (b, len, dim)
            mask = mask.to(device)  # (b, len)
            target = target.to(device)  # (b)

            # Forward pass
            output = model(source, mask)  # (b, 2)
            loss = loss_function(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += accuracy(output, target)
        print(f'train_loss={total_loss / train_data_loader.n_samples:.3f}', end=' ')
        print(f'train_accuracy={total_correct / train_data_loader.n_samples:.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_f_score = 0
            num_iter = 0
            for batch_idx, (source, mask, target) in tqdm(enumerate(valid_data_loader)):
                source = source.to(device)  # (b, len, dim)
                mask = mask.to(device)  # (b, len)
                target = target.to(device)  # (b)

                output = model(source, mask)  # (b, 2)

                total_loss += loss_function(output, target)
                total_correct += accuracy(output, target)
                total_f_score += f_measure(output, target)
                num_iter = batch_idx + 1
        valid_acc = total_correct / valid_data_loader.n_samples
        valid_f_score = total_f_score / num_iter
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.3f}', end=' ')
        print(f'valid_accuracy={valid_acc:.3f}', end=' ')
        print(f'valid_f_score={valid_f_score:.3f}\n')
        if valid_acc > best_acc:
            torch.save(model.state_dict(),
                       os.path.join(config['arguments']['save_path'], f'best_{file_name}.pth'))
            best_acc = valid_acc


if __name__ == '__main__':
    main()
