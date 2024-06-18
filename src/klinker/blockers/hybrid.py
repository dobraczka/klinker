from .composite import BaseAttrTokenCompositeUniqueNameBlocker
import pathlib
from typing import Callable, Optional, Union, List
from .embedding.blocker import EmbeddingBlocker
from .embedding.blockbuilder import EmbeddingBlockBuilder
from class_resolver import HintOrType, OptionalKwargs
from .embedding.deepblocker import DeepBlocker
from ..encoders.deepblocker import DeepBlockerFrameEncoder
from ..encoders import FrameEncoder
from nltk.tokenize import word_tokenize
from ..data.blocks import KlinkerBlockManager
from ..data import generic_upgrade_from_series, NamedVector


class CompositeRelationalDeepBlocker(BaseAttrTokenCompositeUniqueNameBlocker):
    _relation_blocker: DeepBlocker
    _rel_blocker_cls = DeepBlocker

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        frame_encoder: HintOrType[DeepBlockerFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
        use_unique_name: bool = False,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            rel_blocker_kwargs=dict(
                frame_encoder=frame_encoder,
                frame_encoder_kwargs=frame_encoder_kwargs,
                embedding_block_builder=embedding_block_builder,
                embedding_block_builder_kwargs=embedding_block_builder_kwargs,
            ),
            use_unique_name=use_unique_name,
        )
        # set after instatiating seperate blocker to use setter
        self.save = save
        self.force = force
        self.save_dir = save_dir

    @property
    def save(self) -> bool:
        return self._save

    @save.setter
    def save(self, value: bool):
        self._save = value
        self._relation_blocker.save = value

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, value: bool):
        self._force = value
        self._relation_blocker.force = value

    @property
    def save_dir(self) -> Optional[Union[str, pathlib.Path]]:
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value: Optional[Union[str, pathlib.Path]]):
        if value is None:
            self._save_dir = None
            self._relation_blocker.save_dir = None
        else:
            sd = pathlib.Path(value)
            self._save_dir = sd
            self._relation_blocker.save_dir = sd.joinpath("relation")


class CompositeEmbeddingBlocker(BaseAttrTokenCompositeUniqueNameBlocker):
    _relation_blocker: EmbeddingBlocker
    _rel_blocker_cls = EmbeddingBlocker

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        frame_encoder: HintOrType[FrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            rel_blocker_kwargs=dict(
                frame_encoder=frame_encoder,
                frame_encoder_kwargs=frame_encoder_kwargs,
                embedding_block_builder=embedding_block_builder,
                embedding_block_builder_kwargs=embedding_block_builder_kwargs,
            ),
        )


class CompositeLightEABlocker(CompositeEmbeddingBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        depth: int = 2,
        mini_dim: int = 16,
        rel_dim: Optional[int] = None,
        inner_encoder: HintOrType[FrameEncoder] = None,
        inner_encoder_kwargs: OptionalKwargs = None,
        embedding_block_builder: HintOrType[EmbeddingBlockBuilder] = None,
        embedding_block_builder_kwargs: OptionalKwargs = None,
        save: bool = True,
        save_dir: Optional[Union[str, pathlib.Path]] = None,
        force: bool = False,
        top_n_a: Optional[int] = None,
        top_n_r: Optional[int] = None,
    ):
        super().__init__(
            top_n_a=top_n_a,
            top_n_r=top_n_r,
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            frame_encoder="LightEAFrameEncoder",
            frame_encoder_kwargs=dict(
                depth=depth,
                mini_dim=mini_dim,
                rel_dim=rel_dim,
                attribute_encoder=inner_encoder,
                attribute_encoder_kwargs=inner_encoder_kwargs,
                only_use_neighbor_info=True,
            ),
            embedding_block_builder=embedding_block_builder,
            embedding_block_builder_kwargs=embedding_block_builder_kwargs,
        )

    def _filter_emb(self, emb, unique_block_col) -> NamedVector:
        to_remove = unique_block_col.explode().unique().compute()
        wanted = list(set(emb.names) - set(to_remove))
        return emb.subset(wanted)

    def _compute_rel_blocks(
        self, left, right, left_rel, right_rel, unique_blocks
    ) -> KlinkerBlockManager:
        left_reduced, right_reduced = left.concat_values(), right.concat_values()
        left_reduced = generic_upgrade_from_series(left_reduced, reset_index=False)
        right_reduced = generic_upgrade_from_series(right_reduced, reset_index=False)
        left_emb, right_emb = self._relation_blocker._handle_encode(
            left_reduced, right_reduced, left_rel, right_rel
        )
        left_remaining = self._filter_emb(
            left_emb, unique_blocks.blocks[unique_blocks.blocks.columns[0]]
        )
        right_remaining = self._filter_emb(
            right_emb, unique_blocks.blocks[unique_blocks.blocks.columns[1]]
        )
        return self._relation_blocker.embedding_block_builder.build_blocks(
            left=left_remaining,
            right=right_remaining,
            left_name=left.table_name,
            right_name=right.table_name,
        )
