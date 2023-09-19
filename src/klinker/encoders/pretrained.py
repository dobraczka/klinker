import logging
import os
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pystow
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from gensim import downloader as gensim_downloader
from gensim.models import KeyedVectors
from more_itertools import chunked
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from torch import nn
from torch_max_mem import MemoryUtilizationMaximizer
from tqdm.auto import tqdm

from .base import TokenizedFrameEncoder
from ..data import KlinkerDaskFrame
from ..typing import Frame, GeneralVector
from ..utils import (
    concat_frames,
    get_preferred_device,
    resolve_device,
    upgrade_to_sequence,
)

logger = logging.getLogger(__name__)
memory_utilization_maximizer = MemoryUtilizationMaximizer()

word_embedding_dir = pystow.module("klinker").join("word_embeddings")

memory_utilization_maximizer = MemoryUtilizationMaximizer()


# copy-pasted from pykeen
@memory_utilization_maximizer
def _encode_all_memory_utilization_optimized(
    encoder: "TextEncoder",
    labels: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode all labels with the given batch-size.

    Wrapped by memory utilization maximizer to automatically reduce the batch size if needed.

    :param encoder:
        the encoder
    :param labels:
        the labels to encode
    :param batch_size:
        the batch size to use. Will automatically be reduced if necessary.

    :return: shape: `(len(labels), dim)`
        the encoded labels
    """
    return torch.cat(
        [
            encoder(batch)
            for batch in chunked(tqdm(map(str, labels), leave=False), batch_size)
        ],
        dim=0,
    )


class TextEncoder(nn.Module):
    """An encoder for text."""

    def forward(self, labels: Union[str, Sequence[str]]) -> torch.FloatTensor:
        """
        Encode a batch of text.

        :param labels: length: b
            the texts

        :return: shape: `(b, dim)`
            an encoding of the texts
        """
        labels = upgrade_to_sequence(labels)
        labels = list(map(str, labels))
        return self.forward_normalized(texts=labels)

    @abstractmethod
    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:
        """
        Encode a batch of text.

        :param texts: length: b
            the texts

        :return: shape: `(b, dim)`
            an encoding of the texts
        """
        raise NotImplementedError

    @torch.inference_mode()
    def encode_all(
        self,
        labels: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        """Encode all labels (inference mode & batched).

        :param labels:
            a sequence of strings to encode
        :param batch_size:
            the batch size to use for encoding the labels. ``batch_size=1``
            means that the labels are encoded one-by-one, while ``batch_size=len(labels)``
            would correspond to encoding all at once.
            Larger batch sizes increase memory requirements, but may be computationally
            more efficient. `batch_size` can also be set to `None` to enable automatic batch
            size maximization for the employed hardware.

        :returns: shape: (len(labels), dim)
            a tensor representing the encodings for all labels
        """
        return _encode_all_memory_utilization_optimized(
            encoder=self, labels=labels, batch_size=batch_size or len(labels)
        ).detach()


class TransformerTextEncoder(TextEncoder):
    """A combination of a tokenizer and a model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
    ):
        """
        Initialize the encoder using :class:`transformers.AutoModel`.

        :param pretrained_model_name_or_path:
            the name of the pretrained model, or a path, cf. :meth:`transformers.AutoModel.from_pretrained`
        :param max_length: >0, default: 512
            the maximum number of tokens to pad/trim the labels to

        :raises ImportError:
            if the :mod:`transformers` library could not be imported
        """
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
        except ImportError as error:
            raise ImportError(
                "Please install the `transformers` library, use the _transformers_ extra"
                " for PyKEEN with `pip install pykeen[transformers] when installing, or "
                " see the PyKEEN installation docs at https://pykeen.readthedocs.io/en/stable/installation.html"
                " for more information."
            ) from error

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        ).to(resolve_device())
        self.max_length = max_length or 512

    # docstr-coverage: inherited
    def forward_normalized(
        self, texts: Sequence[str]
    ) -> torch.FloatTensor:  # noqa: D102
        return self.model(
            **self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(get_preferred_device(self.model))
        ).pooler_output


class TransformerTokenizedFrameEncoder(TokenizedFrameEncoder):
    encoder: TransformerTextEncoder

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
        batch_size: Optional[int] = None,
    ):
        self.encoder = TransformerTextEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_length=max_length,
        )
        self.batch_size = batch_size

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.encoder.tokenizer.tokenize

    def _encode_side(self, df: Frame) -> GeneralVector:
        return self.encoder.encode_all(
            df[df.columns[0]].values, batch_size=self.batch_size
        )

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self._encode_side(left), self._encode_side(right)


class SentenceTransformerTextEncoder(TransformerTextEncoder):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def forward_normalized(self, texts: Sequence[str]) -> torch.FloatTensor:
        return self.model.encode(texts, convert_to_numpy=False, convert_to_tensor=True)


class SentenceTransformerTokenizedFrameEncoder(TransformerTokenizedFrameEncoder):
    _shortname_mapping = {
        "smpnet": "all-mpnet-base-v2",
        "st5": "gtr-t5-large",
        "sdistilroberta": "all-distilroberta-v1",
        "sminilm": "all-MiniLM-L12-v2",
        "glove": "average_word_embeddings_glove.6B.300d",
    }

    def __init__(self, model_name: str = "st5", batch_size: Optional[int] = None):
        model_name = self.__class__._shortname_mapping.get(model_name, model_name)
        self.encoder = SentenceTransformerTextEncoder(model_name)
        self.batch_size = batch_size


class TokenizedWordEmbedder:
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
        if self._embedding_dim == -1:
            self._embedding_dim = self.embedding_fn("hello").shape[0]
        return self._embedding_dim

    def embed(self, values: str) -> np.ndarray:
        return self.weighted_embed(values, {})

    def weighted_embed(
        self, values: str, weight_mapping: Dict[str, float]
    ) -> np.ndarray:
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
    df: Frame, twe: TokenizedWordEmbedder, weight_dict: Dict = None
) -> np.ndarray:
    embeddings: np.ndarray = torch.nn.init.xavier_normal_(
        torch.empty(len(df), twe.embedding_dim)
    ).numpy()
    # TODO vectorize this?
    for idx, val in enumerate(df[df.columns[0]].values):
        if weight_dict:
            emb = twe.weighted_embed(val, weight_dict)
        else:
            emb = twe.embed(val)
        if not any(np.isnan(emb)):
            embeddings[idx] = emb
    return embeddings


# TODO refactor both classes into TokenEmbeddingAggregator and create AggregatedTokenizedFrameEncoder class
# with tokenized_word_embedder and token_embedding_aggregator
class AverageEmbeddingTokenizedFrameEncoder(TokenizedFrameEncoder):
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
    def __init__(
        self,
        sif_weighting_param=1e-3,
        remove_pc=True,
        min_freq=0,
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
        return self.tokenized_word_embedder.tokenizer_fn

    def prepare(self, left: Frame, right: Frame) -> Tuple[Frame, Frame]:
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
