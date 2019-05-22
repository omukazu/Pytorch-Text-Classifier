import json
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import torch

from miscellaneous.metrics import loss_function, accuracy, f_measure
from miscellaneous.utils import load_setting, create_save_file_name, create_config


def main():
    parser = ArgumentParser(description='train a classifier', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers\nto specify')
    parser.add_argument('--debug', '-d', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    os.makedirs(os.path.dirname(config["arguments"]["save_path"]), exist_ok=True)

    model, device, train_data_loader, valid_data_loader, optimizer = load_setting(config, args)
    file_name = create_save_file_name(config, model.params)
    with open(os.path.join(config["arguments"]["save_path"], f'best_{file_name}.config'), "w") as f:
        json.dump(create_config(config, model.params), f, indent=4)

    best_acc = -1

    for epoch in range(1, config["arguments"]["epoch"] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch_idx, (source, mask, target) in enumerate(train_data_loader):
            source = source.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            output = model(source, mask)
            loss = loss_function(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy(output, target)
        else:
            print(f'train_loss={total_loss / (batch_idx + 1):.3f}', end=' ')
            print(f'train_accuracy={total_accuracy / train_data_loader.n_sample:.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            total_f_score = 0
            for batch_idx, (source, mask, target) in enumerate(valid_data_loader):
                source = source.to(device)
                mask = mask.to(device)
                target = target.to(device)

                output = model(source, mask)

                total_loss += loss_function(output, target)
                total_accuracy += accuracy(output, target)
                total_f_score += f_measure(output, target)
            else:
                valid_acc = total_accuracy / valid_data_loader.n_sample
                valid_f_score = total_f_score / (batch_idx + 1)
                print(f'valid_loss={total_loss / (batch_idx + 1):.3f}', end=' ')
                print(f'valid_accuracy={valid_acc:.3f}', end=' ')
                print(f'valid_f_score={valid_f_score:.3f}\n')
        if valid_acc > best_acc:
            torch.save(model.state_dict(),
                       os.path.join(config["arguments"]["save_path"], f'best_{file_name}.pth'))
            best_acc = valid_acc


if __name__ == '__main__':
    main()
