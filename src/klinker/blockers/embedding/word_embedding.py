import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from gensim import downloader as gensim_downloader
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD

from klinker.typing import GeneralVector

from .encoder import FrameEncoder

logger = logging.getLogger(__name__)


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
                self.embedding_fn = gensim_downloader.load(
                    TokenizedWordEmbedder._gensim_mapping_download[embedding_fn]
                ).__getitem__
            else:
                self.embedding_fn = gensim_downloader.load(embedding_fn).__getitem__
        else:
            self.embedding_fn = embedding_fn
        self.tokenizer_fn = tokenizer_fn

    def embed(self, values: str) -> np.ndarray:
        return np.vstack([self.embedding_fn(tok) for tok in self.tokenizer_fn(values)])

    def weighted_embed(
        self, values: str, weight_mapping: Dict[str, float]
    ) -> np.ndarray:
        return np.vstack(
            [
                self.embedding_fn(tok) * weight_mapping[tok]
                for tok in self.tokenizer_fn(values)
            ]
        )


tokenized_word_embedder_resolver = ClassResolver(
    [TokenizedWordEmbedder], base=TokenizedWordEmbedder, default=TokenizedWordEmbedder
)


class AverageEmbeddingFrameEncoder(FrameEncoder):
    def __init__(
        self,
        tokenized_word_embedder: HintOrType[TokenizedWordEmbedder] = None,
        tokenized_word_embedder_kwargs: OptionalKwargs = None,
    ):
        self.tokenized_word_embedder = tokenized_word_embedder_resolver.make(
            tokenized_word_embedder, tokenized_word_embedder_kwargs
        )

    def _encode_side(self, df: pd.DataFrame) -> GeneralVector:
        return np.array(
            [
                np.mean(
                    self.tokenized_word_embedder.embed(val),
                    axis=0,
                )
                for val in df[df.columns[0]].values
            ]
        )

    def _encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        return (
            self._encode_side(left),
            self._encode_side(right),
        )


class SIFEmbeddingFrameEncoder(FrameEncoder):
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

    def _preprocess(self, left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, float]:
        # use this instead of pd.concat in case columns have different
        # names, we already validated both dfs only have 1 column
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
        return token_weight_dict

    def _encode_side(
        self, df: pd.DataFrame, token_weight_dict: Dict[str, float]
    ) -> GeneralVector:
        embeddings = np.array(
            [
                np.mean(
                    self.tokenized_word_embedder.weighted_embed(val, token_weight_dict),
                    axis=0,
                )
                for val in df[df.columns[0]].values
            ]
        )

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
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        token_weight_dict = self._preprocess(left, right)

        return self._encode_side(left, token_weight_dict), self._encode_side(
            right, token_weight_dict
        )
