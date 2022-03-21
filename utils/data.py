from torch import Tensor, Generator
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from torch import default_generator
from torch.utils.data import Dataset, Subset
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
from torch._utils import _accumulate
from torch import randperm
import torch
def order_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    # indices = randperm(sum(lengths), generator=generator).tolist()
    indices = torch.arange(sum(lengths), dtype=torch.long).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]