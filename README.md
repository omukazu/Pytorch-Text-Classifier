# Pytorch Text Classifier

This repository contains some models for text classification using Pytorch 1.1.0

## Data Preparetion

It is required that data has "labels" and "text" columns and they can be separated by a delimiter such as tab, comma, and so on.

    > 2\tFoo
    > 0\tHoge
    > 1\tHogeHoge
    > ...

## Environment
```
$ pip install pipenv
$ pipenv sync
$ pipenv shell
```

## Model
The following models are available.

1. CNN

![CNN](https://github.com/omukazu/Pytorch-Text-Classifier/blob/images/image/CNN.pdf)

2. BiLSTM with self-attention mechanism

![LSTM](https://github.com/omukazu/Pytorch-Text-Classifier/blob/images/image/LSTM.pdf)

3. The encoder part of Transformer

![Transformer](https://github.com/omukazu/Pytorch-Text-Classifier/blob/images/image/Transformer.pdf)

## Train
You have to create a config file in advance, referring to config/sample.json
And then run the following command.
```
$ python src/train.py config/my_config.json [**kwargs]
```