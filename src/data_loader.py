from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

PAD = 0


class PNDataset(Dataset):
    def __init__(self,
                 path: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int]):
        self.word_to_id = word_to_id
        self.max_seq_len = max_seq_len
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = self.max_seq_len if self.max_seq_len is not None\
            else max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self,
                    idx
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_phrase_len = len(self.sources[idx])
        max_len = self.max_seq_len if self.max_seq_len is not None else self.max_phrase_len
        pad: List[int] = [PAD] * (max_len - max_phrase_len)
        source = np.array(self.sources[idx] + pad)
        mask = np.array([1] * max_phrase_len + [0] * (max_len - max_phrase_len))  # (max_seq_len)
        target = np.array(self.targets[idx])                                      # (1)
        return source, mask, target

    def _load(self,
              path: str,
              delimiter: str = '\t'
              ) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        # tags = ['0', '1', '2', '3']
        tags = ['-1', '1']
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split(delimiter)
                assert tag in tags
                targets.append(tags.index(tag))
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


class PNDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        self.dataset = PNDataset(path, word_to_id, max_seq_len)
        self.n_samples = len(self.dataset)
        super(PNDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
