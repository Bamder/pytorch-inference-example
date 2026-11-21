import numpy
import torch
import math

import numpy.typing as ntyping

from collections.abc import Iterator
from pathlib import Path

def load_byte_stream(
        path : Path, 
        dtype : ntyping.DTypeLike = numpy.uint8,
        shape : tuple[int, ...] = (),
        offset : int = 0,
        read_count : int = -1
    ) -> ntyping.NDArray:
    ''' offset for bytes; shape and read_count for elements'''

    per_record_size = math.prod(shape) if shape else 1
    if read_count > 0 and read_count % per_record_size:
        raise ValueError(f"read_count({read_count}) is not an integer multiple of pre-record size({per_record_size}).")

    buffer = numpy.fromfile(path, dtype=dtype, offset=offset, count=read_count)
    return buffer.reshape(-1, *shape)

def load_set(sample_path : Path, label_path : Path):

    # image set header：4 * int32
    image_magic, image_count, image_rows, image_cols = load_byte_stream(
        path=sample_path, dtype=">i4", shape=(), read_count=4
    ).tolist()

    images = load_byte_stream(
        path=sample_path, dtype=numpy.uint8, shape=(image_rows, image_cols), offset=4 * 4
    )

    # label set header：2 * int32
    label_magic, label_count = load_byte_stream(
        path=label_path, dtype=">i4", shape=(), read_count=2
    ).tolist()

    labels = load_byte_stream(
        path=label_path, dtype=numpy.uint8, shape=(), offset=4 * 2
    )

    return images, labels


class BatchGenerator:

    def __init__(
        self, 
        samples : torch.Tensor, 
        labels : torch.Tensor, 
        batch_size : int, 
        shuffle : bool = True, 
        rand_generator : torch.Generator | None = None
    ):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rand_generator = rand_generator

        self.num_usable = min(self.samples.size(0), self.labels.size(0))

    def __len__(self) -> int:
        # batch number
        return math.ceil(self.num_usable / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:

        # generate random order
        random_order_indices = torch.randperm(self.num_usable, generator=self.rand_generator) if self.shuffle else torch.arange(self.num_usable)
        
        for batch_start_index in range(0, self.num_usable, self.batch_size):
            batch_indices = random_order_indices[batch_start_index:batch_start_index + self.batch_size]
            yield self.samples[batch_indices], self.labels[batch_indices]