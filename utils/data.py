from torch import Tensor, Generator
from typing import TypeVar, List, Optional, Tuple, Sequence
from torch import default_generator
from torch.utils.data import Dataset, Subset
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
from torch._utils import _accumulate
from torch import randperm
import torch
from torch.utils.data import Sampler

class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements Sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, x):
        if hasattr(self.dataset, "inverse_transform"):
            return self.dataset.inverse_transform(x)
        else:
            return x

def order_split(dataset: Dataset[T], lengths: list) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> order_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> order_split(range(10), [3, -1], generator=torch.Generator().manual_seed(42))
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (list): lengths of splits to be produced
    """
    try:
        idx = lengths.index(-1)
        lengths[idx] = len(dataset) - sum(lengths) + 1
    except:
        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):  # type: ignore
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    # indices = randperm(sum(lengths), generator=generator).tolist()
    indices = torch.arange(sum(lengths), dtype=torch.long).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]