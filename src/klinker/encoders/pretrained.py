import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from gensim import downloader as gensim_downloader
from nltk.tokenize import word_tokenize
from pykeen.nn.text import TransformerTextEncoder
from sklearn.decomposition import TruncatedSVD

from ..typing import GeneralVector

from .base import FrameEncoder, TokenizedFrameEncoder

logger = logging.getLogger(__name__)


class TransformerTokenizedFrameEncoder(TokenizedFrameEncoder):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
    ):
        self.encoder = TransformerTextEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_length=max_length,
        )

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.encoder.tokenizer.tokenize

    def _encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self.encoder.encode_all(left.values), self.encoder.encode_all(
            left.values
        )


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

    def prepare(self, left: pd.DataFrame, right: pd.DataFrame):
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
        self.token_weight_dict = token_weight_dict

    def _encode_side(self, df: pd.DataFrame) -> GeneralVector:
        assert self.token_weight_dict is not None
        embeddings = np.array(
            [
                np.mean(
                    self.tokenized_word_embedder.weighted_embed(
                        val, self.token_weight_dict
                    ),
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
        if self.token_weight_dict is None:
            self.prepare(left, right)

        return self._encode_side(left), self._encode_side(right)


frame_encoder_resolver = ClassResolver(
    [
        TransformerTokenizedFrameEncoder,
        AverageEmbeddingTokenizedFrameEncoder,
        SIFEmbeddingTokenizedFrameEncoder,
    ],
    base=FrameEncoder,
    default=SIFEmbeddingTokenizedFrameEncoder,
)

tokenized_frame_encoder_resolver = ClassResolver(
    [
        TransformerTokenizedFrameEncoder,
        AverageEmbeddingTokenizedFrameEncoder,
        SIFEmbeddingTokenizedFrameEncoder,
    ],
    base=TokenizedFrameEncoder,
    default=SIFEmbeddingTokenizedFrameEncoder,
)
