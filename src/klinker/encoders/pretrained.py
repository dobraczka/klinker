import logging
import math
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pystow
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from gensim import downloader as gensim_downloader
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None

from ..data import KlinkerDaskFrame
from ..typing import Frame, GeneralVector
from ..utils import concat_frames
from .base import TokenizedFrameEncoder

logger = logging.getLogger(__name__)

word_embedding_dir = pystow.module("klinker").join("word_embeddings")


def _batch_generator(df, batch_size):
    number_of_batches = math.ceil(len(df) / batch_size)
    start = 0
    if isinstance(df, dd.DataFrame):
        arr = df[df.columns[0]].to_dask_array(lengths=True)
    else:
        arr = df[df.columns[0]].values
    for nb in tqdm(range(number_of_batches), desc="Batches: "):
        start = batch_size * nb
        end = start + batch_size
        if end > len(arr):
            res = arr[start:]
            if isinstance(df, dd.DataFrame):
                yield res.compute()
            else:
                yield res
        res = arr[start:end]
        if isinstance(df, dd.DataFrame):
            yield res.compute()
        else:
            yield res


class TransformerTokenizedFrameEncoder(TokenizedFrameEncoder):
    """Encode frames using pre-trained transformer.

    See <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel.from_pretrained> for more information on pretrained models.

    Args:
    ----
        model_name: str: Transformer name or path
        max_length: int: max number of tokens per row
        batch_size: int: size of batch for encoding

    Examples:
    --------
        >>> # doctest: +SKIP
        >>> import pandas as pd

        >>> from klinker.data import KlinkerPandasFrame
        >>> from klinker.encoders import TransformerTokenizedFrameEncoder

        >>> left = KlinkerPandasFrame.from_df(
                 pd.DataFrame(
                     [("a1", "John Doe"), ("a2", "Jane Doe")], columns=["id", "values"]
                 ),
                 table_name="A",
                 id_col="id",
            ).set_index("id")
        >>> right = KlinkerPandasFrame.from_df(
                pd.DataFrame(
                    [("b1", "Johnny Doe"), ("b2", "Jane Doe")], columns=["id", "values"]
                ),
                table_name="B",
                id_col="id",
            ).set_index("id")
        >>> ttfe = TransformerTokenizedFrameEncoder(
                model_name="bert-base-cased",
                max_length=10,
                batch_size=2
            )
        >>> left_enc, right_enc = ttfe.encode(left=left, right=right)

    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        max_length: int = 128,
        batch_size: int = 512,
    ):
        if AutoModel is None:
            raise ImportError("Please install the transformers library!")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.tokenizer.tokenize

    @torch.no_grad()
    def _encode_side(self, df: Frame) -> GeneralVector:
        encoded = []
        for batch in _batch_generator(df, self.batch_size):
            tok = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded.append(self.model(**tok).pooler_output.detach())
        return torch.vstack(encoded)

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self._encode_side(left), self._encode_side(right)


class SentenceTransformerTokenizedFrameEncoder(TokenizedFrameEncoder):
    """Uses sentencetransformer library to encode frames.

    See <https://www.sbert.net/docs/pretrained_models.html> for a list of models.

    Args:
    ----
        model_name: str: pretrained model name
        max_length: int: max number of tokens per row
        batch_size: int: size of batch for encoding

    Examples:
    --------
        >>> # doctest: +SKIP
        >>> import pandas as pd

        >>> from klinker.data import KlinkerPandasFrame
        >>> from klinker.encoders import SentenceTransformerTokenizedFrameEncoder

        >>> left = KlinkerPandasFrame.from_df(
                 pd.DataFrame(
                     [("a1", "John Doe"), ("a2", "Jane Doe")], columns=["id", "values"]
                 ),
                 table_name="A",
                 id_col="id",
            ).set_index("id")
        >>> right = KlinkerPandasFrame.from_df(
                pd.DataFrame(
                    [("b1", "Johnny Doe"), ("b2", "Jane Doe")], columns=["id", "values"]
                ),
                table_name="B",
                id_col="id",
            ).set_index("id")
        >>> ttfe = SentenceTransformerTokenizedFrameEncoder(
                model_name="st5",
                max_length=10,
                batch_size=2
            )
        >>> left_enc, right_enc = ttfe.encode(left=left, right=right)

    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_length: int = 128,
        batch_size: int = 512,
    ):
        if SentenceTransformer is None:
            raise ImportError("Please install the sentence-transformers library!")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.model.tokenizer.tokenize

    @torch.no_grad()
    def _encode_side(self, df: Frame) -> GeneralVector:
        encoded = []
        for batch in _batch_generator(df, self.batch_size):
            encoded.append(
                self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            )
        return torch.vstack(encoded)

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self._encode_side(left), self._encode_side(right)


class TokenizedWordEmbedder:
    """Encode using pre-trained word embeddings.

    Args:
    ----
      embedding_fn: Union[str, Callable[[str], GeneralVector]]: Either one of "fasttext","glove","word2vec" or embedding function
      tokenizer_fn: Callable[[str], List[str]]: Tokenizer function.
    """

    _gensim_mapping_download = {
        "fasttext": "fasttext-wiki-news-subwords-300",
        "glove": "glove-wiki-gigaword-300",
        "word2vec": "word2vec-google-news-300",
    }

    def __init__(
        self,
        embedding_fn: Union[str, Callable[[str], GeneralVector]] = "fasttext",
        tokenizer_fn: Callable[[str], List[str]] = word_tokenize,
    ):
        # TODO delay loading
        if isinstance(embedding_fn, str):
            if embedding_fn in TokenizedWordEmbedder._gensim_mapping_download:
                actual_name = TokenizedWordEmbedder._gensim_mapping_download[
                    embedding_fn
                ]
                memmap_path = str(word_embedding_dir.joinpath(f"{actual_name}.kv"))
                if not os.path.exists(memmap_path):
                    kv = gensim_downloader.load(actual_name)
                    kv.save(memmap_path)
                else:
                    kv = KeyedVectors.load(memmap_path, mmap="r")
            else:
                kv = gensim_downloader.load(embedding_fn)
            self.embedding_fn = kv.__getitem__
        else:
            self.embedding_fn = embedding_fn
        self.tokenizer_fn = tokenizer_fn
        self._embedding_dim = -1
        self._unknown_token_counter = 0

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension of pretrained word embeddings."""
        if self._embedding_dim == -1:
            self._embedding_dim = self.embedding_fn("hello").shape[0]
        return self._embedding_dim

    def embed(self, values: str) -> np.ndarray:
        """Tokenizes string and returns average of token embeddings.

        Args:
        ----
          values: str: string value to embed.

        Returns:
        -------
            embedding
        """
        return self.weighted_embed(values, {})

    def weighted_embed(
        self, values: str, weight_mapping: Dict[str, float]
    ) -> np.ndarray:
        """Tokenizes string and returns weighted average of token embeddings.

        Args:
        ----
          values: str: string value to embed.
          weight_mapping: Dict[str, float]: weights for tokens.

        Returns:
        -------
            embedding
        """
        # TODO fix code duplication across embed methods can be solved better
        embedded: List[GeneralVector] = []
        for tok in self.tokenizer_fn(values):
            try:
                tok_emb = self.embedding_fn(tok) * weight_mapping.get(tok, 1.0)
                embedded.append(tok_emb)
            except KeyError:
                self._unknown_token_counter += 1
                continue
        if len(embedded) == 0:
            return np.array([np.nan] * self.embedding_dim)
        emb: np.ndarray = np.mean(np.vstack(embedded), axis=0)
        return emb


tokenized_word_embedder_resolver = ClassResolver(
    [TokenizedWordEmbedder], base=TokenizedWordEmbedder, default=TokenizedWordEmbedder
)


def encode_frame(
    df: Frame, twe: TokenizedWordEmbedder, weight_dict: Optional[Dict] = None
) -> np.ndarray:
    """Encode Frame with tokenized word embedder.

    Args:
    ----
      df: Frame:
      twe: TokenizedWordEmbedder:
      weight_dict: Dict:  (Default value = None)

    Returns:
    -------
        embeddings
    """
    embeddings: np.ndarray = torch.nn.init.xavier_normal_(
        torch.empty(len(df), twe.embedding_dim)
    ).numpy()
    # TODO vectorize this?
    for idx, val in enumerate(df[df.columns[0]].values):
        emb = twe.weighted_embed(val, weight_dict) if weight_dict else twe.embed(val)
        if not any(np.isnan(emb)):
            embeddings[idx] = emb
    return embeddings


# TODO refactor both classes into TokenEmbeddingAggregator and create AggregatedTokenizedFrameEncoder class
# with tokenized_word_embedder and token_embedding_aggregator
class AverageEmbeddingTokenizedFrameEncoder(TokenizedFrameEncoder):
    """Averages embeddings of tokenized entity attribute values.

    Args:
    ----
        tokenized_word_embedder: HintOrType[TokenizedWordEmbedder]: Word Embedding class,
        tokenized_word_embedder_kwargs: OptionalKwargs: Keyword arguments for initalizing word embedder
    """

    def __init__(
        self,
        tokenized_word_embedder: HintOrType[TokenizedWordEmbedder] = None,
        tokenized_word_embedder_kwargs: OptionalKwargs = None,
    ):
        self.tokenized_word_embedder = tokenized_word_embedder_resolver.make(
            tokenized_word_embedder, tokenized_word_embedder_kwargs
        )

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.tokenized_word_embedder.tokenizer_fn

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        if isinstance(left, dd.DataFrame):
            left = left.compute()
            right = right.compute()
        return (
            encode_frame(left, twe=self.tokenized_word_embedder),
            encode_frame(right, twe=self.tokenized_word_embedder),
        )


class SIFEmbeddingTokenizedFrameEncoder(TokenizedFrameEncoder):
    """Use Smooth Inverse Frequency weighting scheme to aggregate token embeddings.

    Args:
    ----
        sif_weighting_param: float: weighting parameter
        remove_pc:bool: remove first principal component
        min_freq: int: minimum frequency of occurence
        tokenized_word_embedder: HintOrType[TokenizedWordEmbedder]: Word Embedding class,
        tokenized_word_embedder_kwargs: OptionalKwargs: Keyword arguments for initalizing word embedder

    Quote: Reference
        Arora et. al.,"A Simple but Tough-to-Beat Baseline for Sentence Embeddings", ICLR 2017 <https://openreview.net/pdf?id=SyK00v5xx>
    """

    def __init__(
        self,
        sif_weighting_param: float = 1e-3,
        remove_pc: bool = True,
        min_freq: int = 0,
        tokenized_word_embedder: HintOrType[TokenizedWordEmbedder] = None,
        tokenized_word_embedder_kwargs: OptionalKwargs = None,
    ):
        self.tokenized_word_embedder = tokenized_word_embedder_resolver.make(
            tokenized_word_embedder, tokenized_word_embedder_kwargs
        )

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq
        self.token_weight_dict: Optional[Dict[str, float]] = None

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        """ """
        return self.tokenized_word_embedder.tokenizer_fn

    def prepare(self, left: Frame, right: Frame) -> Tuple[Frame, Frame]:
        """Prepare value counts.

        Args:
        ----
          left: Frame: left attribute frame.
          right: Frame: right attribute frame.

        Returns:
        -------
            left, right
        """
        left, right = super().prepare(left, right)
        merged_col = "merged"
        left.columns = [merged_col]
        right.columns = [merged_col]
        all_values = concat_frames([left, right])

        value_counts = (
            all_values[merged_col]
            .apply(self.tokenized_word_embedder.tokenizer_fn)
            .explode()
            .value_counts()
        )

        def sif_weighting(x, a: float, min_freq: int, total_tokens: int):
            if x >= min_freq:
                return a / (a + x / total_tokens)
            else:
                return 1.0

        total_tokens = value_counts.sum()
        if isinstance(left, KlinkerDaskFrame):
            total_tokens = total_tokens.compute()

        token_weight_dict = value_counts.apply(
            sif_weighting,
            a=self.sif_weighting_param,
            min_freq=self.min_freq,
            total_tokens=total_tokens,
        )

        if isinstance(left, KlinkerDaskFrame):
            token_weight_dict = token_weight_dict.compute()

        self.token_weight_dict = token_weight_dict.to_dict()
        return left, right

    def _postprocess(self, embeddings) -> GeneralVector:
        # From the code of the SIF paper at
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(embeddings)
            pc = svd.components_
            sif_embeddings = embeddings - embeddings.dot(pc.transpose()) * pc
        else:
            sif_embeddings = embeddings
        return sif_embeddings

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        if self.token_weight_dict is None:
            self.prepare(left, right)
        if isinstance(left, KlinkerDaskFrame):
            left_enc = left.map_partitions(
                encode_frame,
                twe=self.tokenized_word_embedder,
                weight_dict=self.token_weight_dict,
            ).compute()
            right_enc = right.map_partitions(
                encode_frame,
                twe=self.tokenized_word_embedder,
                weight_dict=self.token_weight_dict,
            ).compute()
        else:
            left_enc = encode_frame(
                left,
                twe=self.tokenized_word_embedder,
                weight_dict=self.token_weight_dict,
            )
            right_enc = encode_frame(
                right,
                twe=self.tokenized_word_embedder,
                weight_dict=self.token_weight_dict,
            )
        if self.remove_pc:
            left_enc = self._postprocess(left_enc)
            right_enc = self._postprocess(right_enc)
        return left_enc, right_enc


tokenized_frame_encoder_resolver = ClassResolver(
    [
        TransformerTokenizedFrameEncoder,
        SentenceTransformerTokenizedFrameEncoder,
        AverageEmbeddingTokenizedFrameEncoder,
        SIFEmbeddingTokenizedFrameEncoder,
    ],
    base=TokenizedFrameEncoder,
    default=SIFEmbeddingTokenizedFrameEncoder,
)
