import time
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, overload

import dask.dataframe as dd
import numpy as np
import torch
from class_resolver import OptionalKwargs
from class_resolver.contrib.torch import initializer_resolver
from sylloge.id_mapped import id_map_rel_triples
from torch import nn

from klinker.typing import SeriesType

from ..data import NamedVector
from ..typing import Frame, GeneralVector, GeneralVectorLiteral
from ..utils import cast_general_vector


class FrameEncoder:
    """Base class for encoding a KlinkerFrame as embedding."""

    def validate(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ):
        """Check if frames only consist of one column.

        Args:
        ----
          left: Frame: left attributes.
          right: Frame: right attributes.
          left_rel: Optional[Frame]: left relation triples.
          right_rel: Optional[Frame]: right relation triples.

        Raises:
        ------
            ValueError left/right have more than one column.
        """
        if len(left.columns) != 1 or len(right.columns) != 1:
            raise ValueError(
                "Input DataFrames must consist of single column containing all attribute values!"
            )

    def prepare(self, left: Frame, right: Frame) -> Tuple[Frame, Frame]:
        """Prepare for embedding (fill NaNs with empty string).

        Args:
        ----
          left: Frame: left attributes.
          right: Frame: right attributes.

        Returns:
        -------
            left, right
        """
        return left.fillna(""), right.fillna("")

    def _encode(
        self,
        left: Frame,
        right: Frame,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        raise NotImplementedError

    @overload
    def _encode_as(
        self,
        left: Frame,
        right: Frame,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
        return_type: Literal["np"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def _encode_as(
        self,
        left: Frame,
        right: Frame,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
        return_type: Literal["pt"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _encode_as(
        self,
        left: Frame,
        right: Frame,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
        return_type: GeneralVectorLiteral = "pt",
    ) -> Tuple[GeneralVector, GeneralVector]:
        start = time.time()
        left_enc, right_enc = self._encode(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        left_enc = cast_general_vector(left_enc, return_type=return_type)
        right_enc = cast_general_vector(right_enc, return_type=return_type)
        end = time.time()
        self._encoding_time = end - start
        return left_enc, right_enc

    def encode(
        self,
        left: Frame,
        right: Frame,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
        return_type: GeneralVectorLiteral = "pt",
    ) -> Tuple[NamedVector, NamedVector]:
        """Encode dataframes into named vectors.

        Args:
        ----
          left: Frame: left attribute information.
          right: Frame: right attribute information.
          left_rel: Optional[Frame]: left relation triples.
          right_rel: Optional[Frame]: right relation triples.
          return_type: GeneralVectorLiteral:  Either `pt` or `np` to return as pytorch tensor or numpy array.

        Returns:
        -------
            Embeddings of given left/right dataset.
        """
        self.validate(left, right)
        # TODO check if series can't be used everywhere instead
        # of upgrading in prepare
        left, right = self.prepare(left, right)
        left_enc, right_enc = self._encode_as(
            left=left,
            right=right,
            left_rel=left_rel,
            right_rel=right_rel,
            return_type=return_type,
        )
        if isinstance(left, dd.DataFrame):
            left_names = left.index.compute().tolist()
            right_names = right.index.compute().tolist()
        else:
            left_names = left.index.tolist()
            right_names = right.index.tolist()
        return NamedVector(names=left_names, vectors=left_enc), NamedVector(
            names=right_names, vectors=right_enc
        )


class TokenizedFrameEncoder(FrameEncoder):
    """FrameEncoder that uses tokenization of attribute values."""

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        """ """
        raise NotImplementedError


def initialize_and_fill(
    known: NamedVector[torch.Tensor],
    all_names: Union[List[str], Dict[str, int]],
    initializer=nn.init.xavier_normal_,
    initializer_kwargs: OptionalKwargs = None,
) -> NamedVector[torch.Tensor]:
    """Use initalizer and set known values from NamedVector.

    Args:
    ----
      known: NamedVector[torch.Tensor]: Known Embeddings.
      all_names: Union[List[str], Dict[str, int]]: All entity names.
      initializer: Torch initializer.
      initializer_kwargs: Keyword args passed to initializer.

    Returns:
    -------
        Named Vector filled with known values and others from initializer.

    Examples:
    --------
        >>> from klinker.encoders.base import initialize_and_fill
        >>> from klinker.data import NamedVector
        >>> import torch
        >>> known = NamedVector(["e1","e2"], torch.rand(2,2))
        >>> known # doctest: +SKIP
        NamedVector(0|"e1": [0.1697, 0.6516],
                1|"e2": [0.2020, 0.8281],
                dtype=torch.float32)
        >>> initialize_and_fill(known, ["e0","e1","e2","e3"]) # doctest: +SKIP
        NamedVector(0|"e0": [-0.1876,  1.0404],
                1|"e1": [0.1697, 0.6516],
                2|"e2": [0.2020, 0.8281],
                3|"e3": [ 0.0254, -0.8697],
                dtype=torch.float32)

    """
    if not set(known.names).union(set(all_names)) == set(all_names):
        raise ValueError("Known vector must be subset of all_names!")
    initializer = initializer_resolver.lookup(initializer)
    initializer_kwargs = initializer_kwargs or {}

    # get same shape for single vector
    empty = torch.empty_like(known[0])

    # create full lengthy empty matrix with correct shape
    vector = torch.stack([empty] * len(all_names))
    vector = initializer(vector, **initializer_kwargs)
    nv = NamedVector(names=all_names, vectors=vector)
    nv[known.names] = known.vectors
    return nv


def _get_ids(attr: Frame, rel: Frame) -> Set:
    return set(attr.index).union(set(rel["head"])).union(set(rel["tail"]))


class RelationFrameEncoder(FrameEncoder):
    """Base class for Encoders, that also utilize relational information."""

    attribute_encoder: FrameEncoder

    def validate(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ):
        """Ensure relation info is provided and attribute frames consist of single column.

        Args:
        ----
          left: Frame: left attribute information.
          right: Frame: right attribute information.
          left_rel: Optional[Frame]: left relation triples.
          right_rel: Optional[Frame]: right relation triples.

        Raises:
        ------
            ValueError: If attribute frames consist of multiple columns or relational frames are missing.
        """
        super().validate(left=left, right=right)
        if left_rel is None or right_rel is None:
            raise ValueError(f"{self.__class__.__name__} needs left_rel and right_rel!")

    def _encode_rel(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
    ) -> GeneralVector:
        raise NotImplementedError

    @overload
    def _encode_rel_as(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
        return_type: Literal["np"],
    ) -> np.ndarray:
        ...

    @overload
    def _encode_rel_as(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
        return_type: Literal["pt"],
    ) -> torch.Tensor:
        ...

    def _encode_rel_as(
        self,
        rel_triples_left: np.ndarray,
        rel_triples_right: np.ndarray,
        ent_features: NamedVector,
        return_type: GeneralVectorLiteral = "pt",
    ) -> GeneralVector:
        enc = self._encode_rel(
            rel_triples_left=rel_triples_left,
            rel_triples_right=rel_triples_right,
            ent_features=ent_features,
        )
        return cast_general_vector(enc, return_type=return_type)

    def encode(
        self,
        left: SeriesType,
        right: SeriesType,
        *,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
        return_type: GeneralVectorLiteral = "pt",
    ) -> Tuple[NamedVector, NamedVector]:
        """Encode dataframes into named vectors.

        Args:
        ----
          left: Frame: left attribute information.
          right: Frame: right attribute information.
          *:
          left_rel: Optional[Frame]: left relation triples.
          right_rel: Optional[Frame]: right relation triples.
          return_type: GeneralVectorLiteral:  Either `pt` or `np` to return as pytorch tensor or numpy array.

        Returns:
        -------
            Embeddings of given left/right dataset.
        """
        self.validate(left=left, right=right, left_rel=left_rel, right_rel=right_rel)
        left, right = self.prepare(left, right)

        start = time.time()
        # encode attributes
        left_attr_enc, right_attr_enc = self.attribute_encoder.encode(
            left, right, return_type=return_type
        )
        all_attr_enc = left_attr_enc.concat(right_attr_enc)

        # map string based triples to int
        entity_mapping = all_attr_enc.entity_id_mapping
        if isinstance(left_rel, dd.DataFrame):
            left_rel = left_rel.compute()
            right_rel = right_rel.compute()
        rel_triples_left, entity_mapping, rel_mapping = id_map_rel_triples(
            left_rel, entity_mapping=entity_mapping
        )
        rel_triples_right, entity_mapping, rel_mapping = id_map_rel_triples(
            right_rel,
            entity_mapping=entity_mapping,
            rel_mapping=rel_mapping,
        )

        # initialize entity features randomly and replace with
        # attribute features where known
        ent_features = initialize_and_fill(known=all_attr_enc, all_names=entity_mapping)
        left_ids = list(_get_ids(left, left_rel))
        right_ids = list(_get_ids(right, right_rel))

        # encode relations
        features = self._encode_rel_as(
            rel_triples_left=rel_triples_left,
            rel_triples_right=rel_triples_right,
            ent_features=ent_features,
            return_type=return_type,
        )
        named_features = NamedVector(names=entity_mapping, vectors=features)  # type: ignore

        end = time.time()
        self._encoding_time = end - start
        return named_features.subset(list(left_ids)), named_features.subset(
            list(right_ids)
        )
