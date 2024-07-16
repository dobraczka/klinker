import abc
import pandas as pd
import dask.dataframe as dd
from typing import Optional

from ..data import KlinkerBlockManager
from .concat_utils import concat_values
from ..typing import FrameType, SeriesType


class Blocker(abc.ABC):
    """Abstract Blocker class."""

    @abc.abstractmethod
    def assign(
        self,
        left: FrameType,
        right: FrameType,
        left_rel: Optional[FrameType] = None,
        right_rel: Optional[FrameType] = None,
        left_id_col: str = "head",
        right_id_col: str = "head",
        left_table_name: str = "left",
        right_table_name: str = "right",
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
        ----
          left:  Contains entity attribute information of left dataset.
          right: Contains entity attribute information of right dataset.
          left_rel: Contains relational information of left dataset.
          right_rel: Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """


class AttributeConcatBlocker(Blocker):
    """Base class for Blockers that need to concatenate attribute info."""

    @abc.abstractmethod
    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[FrameType] = None,
        right_rel: Optional[FrameType] = None,
        left_id_col: str = "head",
        right_id_col: str = "head",
        left_table_name: str = "left",
        right_table_name: str = "right",
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
        ----
          left: concatenated entity attribute values of left dataset as series.
          right: concatenated entity attribute values of left dataset as series.
          left_rel: Contains relational information of left dataset.
          right_rel: Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """

    def assign(
        self,
        left: FrameType,
        right: FrameType,
        left_rel: Optional[FrameType] = None,
        right_rel: Optional[FrameType] = None,
        left_id_col: str = "head",
        right_id_col: str = "head",
        left_table_name: str = "left",
        right_table_name: str = "right",
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Will concat all entity attribute information before proceeding.

        Args:
        ----
          left: Contains entity attribute information of left dataset.
          right: Contains entity attribute information of right dataset.
          left_rel: Contains relational information of left dataset.
          right_rel: Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        if not isinstance(left, (pd.Series, dd.Series)):
            left_reduced, right_reduced = (
                concat_values(left, id_col=left_id_col),
                concat_values(right, id_col=right_id_col),
            )
        return self._assign(
            left=left_reduced,
            right=right_reduced,
            left_rel=left_rel,
            right_rel=right_rel,
            left_id_col=left_id_col,
            right_id_col=right_id_col,
            left_table_name=left_table_name,
            right_table_name=right_table_name,
        )
