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

try:
    from sentence_transformers import SentenceTransformer, models
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None

try:
    from cuml import UMAP
    from cuml.decomposition import PCA
except ImportError:
    from sklearn.decomposition import PCA
    from umap import UMAP

from ..data import KlinkerDaskFrame
from ..typing import Frame, GeneralVector
from ..utils import concat_frames
from .base import TokenizedFrameEncoder

logger = logging.getLogger(__name__)

word_embedding_dir = pystow.module("klinker").join("word_embeddings")


def _batch_generator(df, batch_size):
    number_of_batches = math.ceil(len(df) / batch_size)
    start = 0
    arr = df[df.columns[0]].values
    for nb in range(number_of_batches):
        start = batch_size * nb
        end = start + batch_size
        if end > len(arr):
            yield arr[start:]
        yield arr[start:end]


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
        model_name: str = "gtr-t5-base",
        max_length: int = 128,
        batch_size: int = 512,
        reduce_dim_to: Optional[int] = None,
        reduce_sample_perc: float = 0.3,
    ):
        if SentenceTransformer is None:
            raise ImportError("Please install the sentence-transformers library!")
        self.model = SentenceTransformer(model_name)
        logger.info("Loaded model")
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.reduce_dim_to = reduce_dim_to
        self.reduce_sample_perc = reduce_sample_perc
        self._added_reduce_layer = False

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.model.tokenizer.tokenize

    @torch.no_grad()
    def _encode_side(self, df: Frame, convert_to_tensor: bool = True) -> GeneralVector:
        vals = df[df.columns[0]].values
        if isinstance(df, KlinkerDaskFrame):
            vals = vals.compute()
        if convert_to_tensor:
            return self.model.encode(
                vals, batch_size=self.batch_size, convert_to_tensor=True
            )
        return self.model.encode(
            vals, batch_size=self.batch_size, convert_to_numpy=True
        )

    def _add_dimensionality_reduction_layer(self, left: Frame, right: Frame):
        # see https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/dimensionality_reduction.py
        logger.info(
            f"Using PCA to output embeddings with dimensionality of {self.reduce_dim_to}. Training on {self.reduce_sample_perc * 100} % of the data."
        )
        lt_embeddings = self._encode_side(
            left.sample(frac=self.reduce_sample_perc), convert_to_tensor=False
        )
        rt_embeddings = self._encode_side(
            right.sample(frac=self.reduce_sample_perc), convert_to_tensor=False
        )
        train_embeddings = np.concatenate([lt_embeddings, rt_embeddings])
        # Compute PCA on the train embeddings matrix
        pca = PCA(n_components=self.reduce_dim_to)
        pca.fit(train_embeddings)
        pca_comp = np.asarray(pca.components_)

        # We add a dense layer to the model, so that it will produce directly embeddings with the new size
        dense = models.Dense(
            in_features=self.model.get_sentence_embedding_dimension(),
            out_features=self.reduce_dim_to,
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        self.model.add_module("dense", dense)
        logger.info(
            f"Done! Added a dense layer with shape ({dense.in_features}, {dense.out_features}) to the model"
        )
        self._added_reduce_layer = True

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        logger.info("Started encode")
        if self.reduce_dim_to and not self._added_reduce_layer:
            self._add_dimensionality_reduction_layer(left, right)
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
        # TODO delay loading, cleanup 100
        if (isinstance(embedding_fn, str) and embedding_fn == "100wiki.en.bin") or (
            isinstance(embedding_fn, str) and embedding_fn == "25wiki.en.bin"
        ):
            import fasttext

            ft = fasttext.load_model(str(word_embedding_dir.joinpath(embedding_fn)))
            self.embedding_fn = ft.get_word_vector
        elif isinstance(embedding_fn, str):
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
        reduce_dim_to: Optional[int] = None,
        umap_n_neighbors: int = 15,
        umap_min_dist: int = 0.1,
    ):
        self.tokenized_word_embedder = tokenized_word_embedder_resolver.make(
            tokenized_word_embedder, tokenized_word_embedder_kwargs
        )

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq
        self.token_weight_dict: Optional[Dict[str, float]] = None
        self.reduce_dim_to = reduce_dim_to
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

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

    def _postprocess(self, left, right) -> Tuple[GeneralVector, GeneralVector]:
        # From the code of the SIF paper at
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            concat_fn = (
                np.concatenate if isinstance(left, np.ndarray) else torch.concatenate
            )
            embeddings = concat_fn([left, right])
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(embeddings)
            pc = svd.components_
            sif_embeddings = embeddings - embeddings.dot(pc.transpose()) * pc
            return sif_embeddings[: len(left)], sif_embeddings[len(left) :]
        return left, right

    def _reduce_dim(
        self, left_emb: GeneralVector, right_emb: GeneralVector
    ) -> Tuple[GeneralVector, GeneralVector]:
        if self.reduce_dim_to:
            initial_dim = left_emb.shape[0]
            if self.reduce_dim_to == initial_dim:
                logger.info(
                    f"Can't reduce to the same dimensionality ({initial_dim}) so returning"
                )
                return left_emb, right_emb
            if self.reduce_dim_to > initial_dim:
                raise ValueError(
                    f"Cannot reduce embeddings of dimensionality {initial_dim} to higher dimensionality of {self.reduce_dim_to}!"
                )
            logger.info(f"Reducing embedding dim to {self.reduce_dim_to}")
            umap = UMAP(
                n_components=self.reduce_dim_to,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
            )
            all_vec = (
                np.concatenate([left_emb, right_emb])
                if isinstance(left_emb, np.ndarray)
                else torch.concat([left_emb, right_emb])
            )
            reduced_vec = umap.fit_transform(all_vec)
            left_emb = reduced_vec[: len(left_emb)]
            right_emb = reduced_vec[len(left_emb) :]
        return left_emb, right_emb

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
            left_enc, right_enc = self._postprocess(left_enc, right_enc)
        return self._reduce_dim(left_enc, right_enc)


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
