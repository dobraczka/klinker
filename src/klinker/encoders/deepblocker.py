import random
from typing import Callable, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import torch
from class_resolver import ClassResolver, HintOrType, OptionalKwargs
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from klinker.utils import resolve_device

from ..data import KlinkerDaskFrame
from ..models.deepblocker import (
    AutoEncoderDeepBlockerModelTrainer,
    CTTDeepBlockerModelTrainer,
    DeepBlockerModelTrainer,
)
from ..typing import Frame, GeneralVector
from .base import TokenizedFrameEncoder
from .pretrained import tokenized_frame_encoder_resolver

FeatureType = TypeVar("FeatureType")


class DeepBlockerFrameEncoder(Generic[FeatureType], TokenizedFrameEncoder):
    """Base class for DeepBlocker Frame encoders.

    Args:
    ----
        hidden_dimensions: Tuple[int, int]: Hidden dimensions
        num_epochs: int: Number of epochs if training
        batch_size: int: Batch size
        learning_rate: float: Learning rate if training
        loss_function: Optional[_Loss]: Loss function if training
        optimizer: Optional[HintOrType[Optimizer]]: Optimizer if training
        optimizer_kwargs: OptionalKwargs: Keyword arguments to inizialize optimizer
        frame_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        frame_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing frame encoder
    """

    inner_encoder: TokenizedFrameEncoder

    def __init__(
        self,
        hidden_dimensions: Tuple[int, int],
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        loss_function: Optional[_Loss] = None,
        optimizer: Optional[HintOrType[Optimizer]] = None,
        optimizer_kwargs: OptionalKwargs = None,
        frame_encoder: HintOrType[TokenizedFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        self.inner_encoder = tokenized_frame_encoder_resolver.make(
            frame_encoder, frame_encoder_kwargs
        )
        self.hidden_dimensions = hidden_dimensions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dimension: Optional[int] = None
        self.loss_function = loss_function
        self._optimizer_hint = optimizer
        self._optimizer_kwargs = optimizer_kwargs

    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        return self.inner_encoder.tokenizer_fn

    @property
    def trainer_cls(self) -> Type[DeepBlockerModelTrainer[FeatureType]]:
        raise NotImplementedError

    def create_features(
        self, left: Frame, right: Frame
    ) -> Tuple[FeatureType, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        features, left_enc, right_enc = self.create_features(left, right)
        assert self.input_dimension is not None
        assert self.hidden_dimensions is not None
        trainer = self.trainer_cls(
            input_dimension=self.input_dimension,
            hidden_dimensions=self.hidden_dimensions,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            optimizer=self._optimizer_hint,
            optimizer_kwargs=self._optimizer_kwargs,
        )
        device = resolve_device()
        self.model = trainer.train(
            features,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            device=device,
        )
        return self.model.encode_side(left_enc, device), self.model.encode_side(
            right_enc, device
        )


class AutoEncoderDeepBlockerFrameEncoder(DeepBlockerFrameEncoder[torch.Tensor]):
    """Autoencoder class for DeepBlocker Frame encoders.

    Args:
    ----
        hidden_dimensions: Tuple[int, int]: Hidden dimensions
        num_epochs: int: Number of epochs if training
        batch_size: int: Batch size
        learning_rate: float: Learning rate if training
        loss_function: Optional[_Loss]: Loss function if training
        optimizer: Optional[HintOrType[Optimizer]]: Optimizer if training
        optimizer_kwargs: OptionalKwargs: Keyword arguments to inizialize optimizer
        frame_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        frame_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing frame encoder
    """

    def __init__(
        self,
        hidden_dimensions: Tuple[int, int] = (2 * 150, 150),
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        frame_encoder: HintOrType[TokenizedFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        **kwargs,
    ):
        super().__init__(
            hidden_dimensions=hidden_dimensions,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            frame_encoder=frame_encoder,
            frame_encoder_kwargs=frame_encoder_kwargs,
            **kwargs,
        )
        self._input_dimension = -1

    @property
    def trainer_cls(self) -> Type[DeepBlockerModelTrainer[torch.Tensor]]:
        return AutoEncoderDeepBlockerModelTrainer

    def create_features(
        self, left: Frame, right: Frame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Features for AutoEncoder.

        Args:
        ----
          left: Frame: left attributes.
          right: Frame: right attributes.

        Returns:
        -------
            Concatenated left/right encoded, left encoded, right encoded
        """
        left_enc, right_enc = self.inner_encoder._encode_as(
            left, right, return_type="pt"
        )
        left_enc = left_enc.float()
        right_enc = right_enc.float()

        self.input_dimension = left_enc.shape[1]
        return (
            torch.concat([left_enc, right_enc]),
            left_enc,
            right_enc,
        )


class CrossTupleTrainingDeepBlockerFrameEncoder(DeepBlockerFrameEncoder):
    """CrossTupleTraining class for DeepBlocker Frame encoders.

    Args:
    ----
        hidden_dimensions: Tuple[int, int]: Hidden dimensions
        num_epochs: int: Number of epochs
        batch_size: int: Batch size
        learning_rate: float: Learning rate
        synth_tuples_per_tuple: int: Synthetic tuples per tuple
        pos_to_neg_ratio: float: Ratio of positiv to negative tuples
        max_perturbation:float: Degree how much tuples should be corrupted
        random_seed: Seed to control randomness
        loss_function: Optional[_Loss]: Loss function if training
        optimizer: Optional[HintOrType[Optimizer]]: Optimizer if training
        optimizer_kwargs: OptionalKwargs: Keyword arguments to inizialize optimizer
        frame_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        frame_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing frame encoder
    """

    def __init__(
        self,
        hidden_dimensions: Tuple[int, int] = (2 * 150, 150),
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        synth_tuples_per_tuple: int = 5,
        pos_to_neg_ratio: float = 1.0,
        max_perturbation: float = 0.4,
        random_seed=None,
        loss_function: Optional[_Loss] = None,
        optimizer: Optional[HintOrType[Optimizer]] = None,
        optimizer_kwargs: OptionalKwargs = None,
        frame_encoder: HintOrType[TokenizedFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
    ):
        super().__init__(
            hidden_dimensions=hidden_dimensions,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            frame_encoder=frame_encoder,
            frame_encoder_kwargs=frame_encoder_kwargs,
            loss_function=loss_function,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation
        self.random_seed = random_seed

    def create_features(
        self, left: Frame, right: Frame
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """Create features for cross-tuple training.

        Args:
        ----
          left: Frame: left attributes.
          right: Frame: right attributes.

        Returns:
        -------
            (left_training, right_training, labels), left encoded, right encoded
        """
        if isinstance(left, KlinkerDaskFrame):
            raise NotImplementedError(
                "CrossTupleTrainingDeepBlockerFrameEncoder has not been implemented for dask yet!"
            )

        # TODO refactor this function (copy-pasted from deepblocker repo)
        list_of_tuples = pd.DataFrame(
            np.concatenate([left.values, right.values]), columns=["merged"]
        )["merged"]
        num_positives_per_tuple = self.synth_tuples_per_tuple
        num_negatives_per_tuple = int(
            self.synth_tuples_per_tuple * self.pos_to_neg_ratio
        )
        num_tuples = len(list_of_tuples)
        total_number_of_elems = int(
            num_tuples * (num_positives_per_tuple + num_negatives_per_tuple)
        )

        # We create three lists containing T, T' and L respectively
        # We use the following format: first num_tuples * num_positives_per_tuple correspond to T
        # and the remaining correspond to T'
        left_tuple_list = ["" for _ in range(total_number_of_elems)]
        right_tuple_list = ["" for _ in range(total_number_of_elems)]
        label_list = [0 for _ in range(total_number_of_elems)]

        random.seed(self.random_seed)

        tokenizer = self.inner_encoder.tokenizer_fn
        for index in range(len(list_of_tuples)):
            tokenized_tuple = tokenizer(list_of_tuples[index])
            max_tokens_to_remove = int(len(tokenized_tuple) * self.max_perturbation)

            training_data_index = index * (
                num_positives_per_tuple + num_negatives_per_tuple
            )

            # Create num_positives_per_tuple tuple pairs with positive label
            for _ in range(num_positives_per_tuple):
                tokenized_tuple_copy = tokenized_tuple[:]

                # If the tuple has 10 words and max_tokens_to_remove is 0.5, then we can remove at most 5 words
                # we choose a random number between 0 and 5.
                # suppose it is 3. Then we randomly remove 3 words
                num_tokens_to_remove = random.randint(0, max_tokens_to_remove)
                for _ in range(num_tokens_to_remove):
                    # randint is inclusive. so randint(0, 5) can return 5 also
                    tokenized_tuple_copy.pop(
                        random.randint(0, len(tokenized_tuple_copy) - 1)
                    )

                left_tuple_list[training_data_index] = list_of_tuples[index]
                right_tuple_list[training_data_index] = " ".join(tokenized_tuple_copy)
                label_list[training_data_index] = 1
                training_data_index += 1

            for _ in range(num_negatives_per_tuple):
                left_tuple_list[training_data_index] = list_of_tuples[index]
                right_tuple_list[training_data_index] = random.choice(list_of_tuples)
                label_list[training_data_index] = 0
                training_data_index += 1
        left_train_enc, right_train_enc = self.inner_encoder._encode_as(
            pd.DataFrame(left_tuple_list),
            pd.DataFrame(right_tuple_list),
            return_type="pt",
        )
        self.input_dimension = left_train_enc.shape[1]

        left_enc, right_enc = self.inner_encoder._encode_as(
            left, right, return_type="pt"
        )
        return (
            (left_train_enc.float(), right_train_enc.float(), torch.tensor(label_list)),
            left_enc.float(),
            right_enc.float(),
        )

    def _encode(
        self,
        left: Frame,
        right: Frame,
        left_rel: Optional[Frame] = None,
        right_rel: Optional[Frame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        self.inner_encoder.prepare(left, right)
        (
            (left_train, right_train, label_list),
            left_enc,
            right_enc,
        ) = self.create_features(left, right)

        assert self.input_dimension is not None
        trainer = CTTDeepBlockerModelTrainer(
            input_dimension=self.input_dimension,
            hidden_dimensions=self.hidden_dimensions,
            learning_rate=self.learning_rate,
        )
        features = (left_train, right_train, torch.tensor(label_list))
        device = resolve_device()
        self.ctt_model = trainer.train(
            features=features,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            device=device,
        )

        return self.ctt_model.encode_side(left_enc, device), self.ctt_model.encode_side(
            right_enc, device
        )


class HybridDeepBlockerFrameEncoder(CrossTupleTrainingDeepBlockerFrameEncoder):
    """Hybrid DeepBlocker class.

    Uses both Autoencoder and CrossTupleTraining strategy.

    Args:
    ----
        frame_encoder: HintOrType[TokenizedFrameEncoder]: Base encoder class
        frame_encoder_kwargs: OptionalKwargs: Keyword arguments for initializing frame encoder
        hidden_dimensions: Tuple[int, int]: Hidden dimensions
        num_epochs: int: Number of epochs if training
        batch_size: int: Batch size
        learning_rate: float: Learning rate
        synth_tuples_per_tuple: int: Synthetic tuples per tuple
        pos_to_neg_ratio: float: Ratio of positiv to negative tuples
        max_perturbation:float: Degree how much tuples should be corrupted
        random_seed: Seed to control randomness
        loss_function: Optional[_Loss]: Loss function if training
        optimizer: Optional[HintOrType[Optimizer]]: Optimizer if training
        optimizer_kwargs: OptionalKwargs: Keyword arguments to inizialize optimizer
    """

    def __init__(
        self,
        frame_encoder: HintOrType[TokenizedFrameEncoder] = None,
        frame_encoder_kwargs: OptionalKwargs = None,
        hidden_dimensions: Tuple[int, int] = (2 * 150, 150),
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        synth_tuples_per_tuple: int = 5,
        pos_to_neg_ratio: float = 1.0,
        max_perturbation=0.4,
        random_seed=None,
        loss_function: Optional[_Loss] = None,
        optimizer: Optional[HintOrType[Optimizer]] = None,
        optimizer_kwargs: OptionalKwargs = None,
    ):
        inner_encoder = AutoEncoderDeepBlockerFrameEncoder(
            frame_encoder=frame_encoder,
            frame_encoder_kwargs=frame_encoder_kwargs,
            hidden_dimensions=hidden_dimensions,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        super().__init__(
            frame_encoder=inner_encoder,
            hidden_dimensions=hidden_dimensions,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            synth_tuples_per_tuple=synth_tuples_per_tuple,
            pos_to_neg_ratio=pos_to_neg_ratio,
            max_perturbation=max_perturbation,
            random_seed=random_seed,
            loss_function=loss_function,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )


deep_blocker_encoder_resolver = ClassResolver(
    [
        AutoEncoderDeepBlockerFrameEncoder,
        CrossTupleTrainingDeepBlockerFrameEncoder,
        HybridDeepBlockerFrameEncoder,
    ],
    base=DeepBlockerFrameEncoder,
    default=AutoEncoderDeepBlockerFrameEncoder,
)
