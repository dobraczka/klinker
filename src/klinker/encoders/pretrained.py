import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pystow
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from gensim import downloader as gensim_downloader
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from pykeen.nn.text import TransformerTextEncoder
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from torch_max_mem import MemoryUtilizationMaximizer

from .base import TokenizedFrameEncoder
from ..typing import Frame, GeneralVector

logger = logging.getLogger(__name__)
memory_utilization_maximizer = MemoryUtilizationMaximizer()

word_embedding_dir = pystow.module("klinker").join("word_embeddings")


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

    @staticmethod
    def _encode_side(df: Frame, twe: TokenizedWordEmbedder) -> GeneralVector:
        embeddings: np.ndarray = torch.nn.init.xavier_normal_(
            torch.empty(len(df), twe.embedding_dim)
        ).numpy()
        # TODO vectorize this?
        for idx, val in enumerate(df[df.columns[0]].values):
            emb = twe.embed(val)
            if not any(np.isnan(emb)):
                embeddings[idx] = emb
        return embeddings

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        if isinstance(left, dd.DataFrame):
            return (
                left.map_partitions(
                    AverageEmbeddingTokenizedFrameEncoder._encode_side,
                    twe=self.tokenized_word_embedder,
                ).compute(),
                right.map_partitions(
                    AverageEmbeddingTokenizedFrameEncoder._encode_side,
                    twe=self.tokenized_word_embedder,
                ).compute(),
            )

        return (
            AverageEmbeddingTokenizedFrameEncoder._encode_side(
                left, twe=self.tokenized_word_embedder
            ),
            self._encode_side(right, twe=self.tokenized_word_embedder),
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
        # use this instead of pd.concat in case columns have different
        # names, we already validated both dfs only have 1 column
        left, right = super().prepare(left, right)
        merged_col = "merged"
        all_values = pd.DataFrame(
            np.concatenate([left.values, right.values]), columns=[merged_col]
        )

        value_counts = (
            all_values[merged_col]
            .apply(self.tokenized_word_embedder.tokenizer_fn)
            .explode()
            .value_counts()
        )

        total_tokens = value_counts.sum()

        token_weight_dict = {}
        a = self.sif_weighting_param
        for word, frequency in value_counts.items():
            if frequency >= self.min_freq:
                token_weight_dict[word] = a / (a + frequency / total_tokens)
            else:
                token_weight_dict[word] = 1.0
        self.token_weight_dict = token_weight_dict
        return left, right

    def _encode_side(self, df: Frame) -> GeneralVector:
        assert self.token_weight_dict is not None
        embeddings: np.ndarray = torch.nn.init.xavier_normal_(
            torch.empty(len(df), self.tokenized_word_embedder.embedding_dim)
        ).numpy()
        # TODO vectorize this?
        for idx, val in enumerate(df[df.columns[0]].values):
            emb = self.tokenized_word_embedder.weighted_embed(
                val, self.token_weight_dict
            )
            if not any(np.isnan(emb)):
                embeddings[idx] = emb

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
        return self._encode_side(left), self._encode_side(right)


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
