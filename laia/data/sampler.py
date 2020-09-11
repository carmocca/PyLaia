import random
from typing import Callable, Iterable, List

import torch

import laia.common.logging as log

_logger = log.get_logger(__name__)


class SortedSampler:
    def __init__(
        self,
        data: torch.utils.data.Dataset,
        value_fn: Callable,
        reverse: bool = False,
        batch_size: int = 1,
        drop_last: bool = False,
    ):
        _logger.info(f"Preparing {self.__class__.__name__} iterator")
        self.batches = self.prepare_batches(
            data, value_fn, reverse, batch_size, drop_last
        )
        _logger.info(f"{self.__class__.__name__} iterator created")
        num_batches, leftover = divmod(len(data), batch_size)
        self.length = num_batches + (bool(leftover) if not drop_last else 0)

    @staticmethod
    def prepare_batches(
        data, value_fn, reverse, batch_size, drop_last
    ) -> List[List[int]]:
        idxs = sorted(
            range(len(data)), key=lambda i: value_fn(data[i]), reverse=reverse
        )
        # split into batches with the same length
        batches = [idxs[i : i + batch_size] for i in range(0, len(idxs), batch_size)]
        # drop last if necessary
        if drop_last and len(batches[-1]) < batch_size:
            del batches[-1]
        return batches

    def __iter__(self) -> Iterable[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return self.length


class BalancedPaddingSampler:
    def __init__(
        self,
        data: torch.utils.data.Dataset,
        value_fn: Callable,
        reverse: bool = False,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.shuffle = shuffle

        _logger.info(f"Preparing {self.__class__.__name__} iterator...")
        self.batches = self.prepare_batches(data, value_fn, reverse, batch_size)
        _logger.info(f"{self.__class__.__name__} iterator created")
        self.length = len(self.batches)

    @staticmethod
    def prepare_batches(data, value_fn, reverse, batch_size) -> List[List[int]]:
        idxs = sorted(
            range(len(data)), key=lambda i: value_fn(data[i]), reverse=reverse
        )
        # first batch has `batch_size` elements.
        # rest have as many elements as necessary
        # to match the padded size of the first batch
        batches = [idxs[:batch_size]]
        max_size = value_fn(data[idxs[0]]) * batch_size
        i = batch_size
        for j in range(batch_size + 1, len(idxs)):
            size = value_fn(data[idxs[i]]) * (j - i)
            if size >= max_size:
                end = j if size == max_size else j - 1
                batches.append(idxs[i:end])
                i = end
        if idxs[i:]:
            batches.append(idxs[i:])
        return batches

    def __iter__(self) -> Iterable[List[int]]:
        # shuffle batch order
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        return self.length
