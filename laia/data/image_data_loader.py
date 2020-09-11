from typing import Callable, Dict, Optional

import torch

from laia.data.padding_collater import PaddingCollater
from laia.data.sampler import BalancedPaddingSampler, SortedSampler


def by_width(x: Dict) -> int:
    return x["img"].size(2)


def by_numel(x: Dict) -> int:
    return x["img"].numel()


class ImageDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        image_channels: Optional[int] = 1,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        worker_init_fn: Callable = None,
        sampler: Optional[str] = None,
    ):
        if not sampler:
            kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "drop_last": drop_last,
            }
        elif sampler == "sorted":
            kwargs = {
                "batch_sampler": SortedSampler(
                    dataset,
                    value_fn=by_numel,
                    batch_size=batch_size,
                    drop_last=drop_last,
                    reverse=True,
                )
            }
        elif sampler == "balanced":
            kwargs = {
                "batch_sampler": BalancedPaddingSampler(
                    dataset,
                    value_fn=by_numel,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    reverse=True,
                )
            }
        else:
            raise ValueError
        super(ImageDataLoader, self).__init__(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=PaddingCollater(
                {"img": [image_channels, image_height, image_width]},
                # note: sorting the batch dramatically improves the performance
                sort_fn=by_width,
                reverse=True,
            ),
            **kwargs
        )
