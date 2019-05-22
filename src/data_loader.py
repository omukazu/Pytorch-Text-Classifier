from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from miscellaneous.constants import PAD


class MyDataset(Dataset):
    def __init__(self,
                 path: str,
                 labels: List[str],
                 delimiter: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int]
                 ) -> None:
        self.word_to_id = word_to_id
        self.max_seq_len = max_seq_len
        self.sources, self.targets = self._load(path, labels, delimiter)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self,
                    idx: int
                    ) -> Tuple[List, List, int]:
        length = len(self.sources[idx])
        source = self.sources[idx]
        mask = [1] * length
        target = self.targets[idx]
        return source, mask, target

    def _load(self,
              path: str,
              labels: List[str],
              delimiter: str,
              ) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                label, body = line.strip().split(delimiter)
                assert label in labels
                targets.append(labels.index(label))
                ids: List[int] = []
                for mrph in body.split():
                    if mrph in self.word_to_id.keys():
                        ids.append(self.word_to_id[mrph])
                    else:
                        ids.append(self.word_to_id['<UNK>'])
                if self.max_seq_len is not None and len(ids) > self.max_seq_len:
                    ids = ids[-self.max_seq_len:]  # limit sequence length from end of a sentence
                sources.append(ids)
        assert len(sources) == len(targets)
        return sources, targets


class MyDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 labels: List[str],
                 delimiter: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ) -> None:
        self.dataset = MyDataset(path, labels, delimiter, word_to_id, max_seq_len)
        self.n_sample = len(self.dataset)
        super(MyDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=my_collate_fn)


def my_collate_fn(batch: List[Tuple[List, List, int]]
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources, masks, targets = [], [], []
    max_seq_len_in_batch = max(len(sample[0]) for sample in batch)
    for sample in batch:
        source, mask, target = sample
        source_length = len(source)
        source_padding = [PAD] * (max_seq_len_in_batch - source_length)
        source_mask_padding = [0] * (max_seq_len_in_batch - source_length)
        sources.append(source+source_padding)
        masks.append(mask+source_mask_padding)
        targets.append(target)
    return torch.LongTensor(sources), torch.LongTensor(masks), torch.LongTensor(targets)
