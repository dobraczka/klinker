from typing import Callable, Dict, Iterable, List, Optional, Tuple

import dask.dataframe as dd
import pandas as pd
from datasketch import LeanMinHash, MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from klinker.typing import SeriesType

from ..data import (
    KlinkerBlockManager,
    KlinkerFrame,
)
from .base import SchemaAgnosticBlocker
from nltk.corpus import stopwords


# TODO handle code duplication
class FilteredTokenizer:
    def __init__(self, tokenize_fn=None, min_token_length=3, stop_words=None):
        if not tokenize_fn:
            tokenize_fn = word_tokenize
        if not stop_words:
            stop_words = stopwords.words("english")
        self.tokenize_fn = tokenize_fn
        self.stop_words = stop_words
        self.min_token_length = min_token_length

    def tokenize(self, x, return_set: bool = False):
        tokens = filter(
            lambda tok: len(tok) >= self.min_token_length
            and tok not in self.stop_words,
            self.tokenize_fn(str(x.lower())),
        )
        if return_set:
            return set(tokens)
        return list(tokens)


def _insert(
    ser_part: pd.Series, lsh: MinHashLSH, encode_fn: Callable[[str], Iterable[bytes]]
):
    """Insert encoded entity info into MinHashLSH instance.

    Args:
    ----
      ser_part: pd.Series: Series containing concatenated entity attribute values.
      lsh: MinHashLSH: lsh instance where encoded attribues will be inserted.
      encode_fn: Callable[[str],Iterable[bytes]]: encoding function

    Returns:
    -------
        index of given series
    """
    with lsh.insertion_session() as session:
        for key, val in ser_part.items():
            msh = MinHash(num_perm=lsh.h)
            msh.update_batch(encode_fn(val))
            final_msh = LeanMinHash(msh)
            session.insert(key, final_msh)
    return ser_part.index


def _query(
    ser_part: pd.Series,
    lsh: MinHashLSH,
    encode_fn: Callable[[str], Iterable[bytes]],
    left_name: str,
    right_name: str,
):
    """Query the given lsh with the provided entity information.

    Args:
    ----
      ser_part: pd.Series: series holding concatenated entity attribute values.
      lsh: MinHashLSH: filled lsh instance.
      encode_fn: Callable[[str],Iterable[bytes]]: encoding function
      left_name: str: Name of left dataset.
      right_name: str: Name of right dataset.

    Returns:
    -------
        dataframe of blocks
    """
    cur_block: Dict[str, List] = {left_name: [], right_name: []}
    index = []
    for key, val in ser_part.items():
        msh = MinHash(num_perm=lsh.h)
        msh.update_batch(encode_fn(val))
        final_msh = LeanMinHash(msh)
        res = lsh.query(final_msh)
        if len(res) > 0:
            cur_block[left_name].append(list(res))
            cur_block[right_name].append([key])
            index.append(key)
    return pd.DataFrame(cur_block, index=index)


class MinHashLSHBlocker(SchemaAgnosticBlocker):
    """Blocker relying on MinHashLSH procedure.

    Args:
    ----
        tokenize_fn Callable: Function that tokenizes entity attribute values.
        threshold: float: Jaccard threshold to use in underlying lsh procedure.
        num_perm: int: number of permutations used in minhash algorithm.
        weights: Tuple[float,float]: false positive/false negative weighting (must add up to one)

    Attributes:
    ----------
        tokenize_fn Callable: Function that tokenizes entity attribute values.
        threshold: float: Jaccard threshold to use in underlying lsh procedure.
        num_perm: int: number of permutations used in minhash algorithm.
        weights: Tuple[float,float]: false positive/false negative weighting (must add up to one)

    Examples:
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import MinHashLSHBlocker
        >>> blocker = MinHashLSHBlocker(threshold=0.8, weights=(0.7,0.3))
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    """

    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self.tokenizer = FilteredTokenizer(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            stop_words=stop_words,
        )
        self.threshold = threshold
        self.num_perm = num_perm
        self.weights = weights

    def _inner_encode(self, val: str):
        """Encodes string to list of bytes.

        Args:
        ----
          val: str: input string.

        Returns:
        -------
            list of bytes.
        """
        return [tok.encode("utf-8") for tok in self.tokenizer.tokenize(str(val))]

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Uses minhash algorithm to encode entities via tokenized attributes.
        Fills a lsh instance with the left hashes.
        Queries using the right hashes.

        Args:
        ----
          left: SeriesType: concatenated entity attribute values of left dataset as series.
          right: SeriesType: concatenated entity attribute values of left dataset as series.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            weights=self.weights,
        )
        if isinstance(left, dd.Series):
            left.map_partitions(
                _insert,
                lsh=lsh,
                encode_fn=self._inner_encode,
                meta=left._meta.index,
            ).compute()
            blocks = right.map_partitions(
                _query,
                lsh=lsh,
                encode_fn=self._inner_encode,
                left_name=left.name,
                right_name=right.name,
                meta=pd.DataFrame([], columns=[left.name, right.name], dtype="O"),
            )
            return KlinkerBlockManager(blocks)
        else:
            _insert(left, lsh=lsh, encode_fn=self._inner_encode)
            blocks = _query(
                right,
                lsh=lsh,
                encode_fn=self._inner_encode,
                left_name=left.name,
                right_name=right.name,
            )
            return KlinkerBlockManager.from_pandas(blocks)
