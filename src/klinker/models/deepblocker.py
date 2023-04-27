import os
from abc import ABC, abstractmethod
from typing import IO, BinaryIO, Generic, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs
from class_resolver.contrib.torch import optimizer_resolver
from pykeen.utils import resolve_device
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

FeatureType = TypeVar("FeatureType")


class FeatureDataset(Dataset[torch.Tensor]):
    def __init__(self, features: torch.Tensor):
        self.features = features

    def __getitem__(self, index) -> torch.Tensor:
        return self.features[index].float()

    def __len__(self):
        return len(self.features)


class TripletDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor):
        assert all(
            left.size(0) == tensor.size(0) for tensor in [right, labels]
        ), "Size mismatch between tensors"
        self.left = left.float()
        self.right = right.float()
        self.labels = labels.float()

    def __getitem__(self, index):
        return self.left[index], self.right[index], self.labels[index]

    def __len__(self):
        return self.left.size(0)


class DeepBlockerModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimensions: Tuple[int, int]):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def encode_side(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class AutoEncoderDeepBlockerModel(DeepBlockerModel):
    def __init__(self, input_dimension: int, hidden_dimensions: Tuple[int, int]):
        super().__init__(
            input_dimension=input_dimension, hidden_dimensions=hidden_dimensions
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dimensions[0], self.hidden_dimensions[1]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dimensions[1], self.hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dimensions[0], self.input_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode_side(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.encoder(x).detach().numpy()


class CTTDeepBlockerModel(DeepBlockerModel):
    def __init__(self, input_dimension: int, hidden_dimensions: Tuple[int, int]):
        super().__init__(
            input_dimension=input_dimension, hidden_dimensions=hidden_dimensions
        )
        self.siamese_summarizer = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dimensions[0], self.hidden_dimensions[1]),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(self.hidden_dimensions[1], 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.siamese_summarizer(x1)
        x2 = self.siamese_summarizer(x2)
        pred = self.classifier(torch.abs(x1 - x2))
        return torch.sigmoid(pred)

    def encode_side(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.siamese_summarizer(x).detach().numpy()


class DeepBlockerModelTrainer(Generic[FeatureType], ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Tuple[int, int],
        learning_rate: float,
        loss_function: _Loss = None,
        optimizer: HintOrType[Optimizer] = None,
        optimizer_kwargs: OptionalKwargs = None,
    ):
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = self.model_cls(
            input_dimension=input_dimension, hidden_dimensions=hidden_dimensions
        )
        optimizer_kwargs = (
            {"lr": self.learning_rate} if optimizer_kwargs is None else optimizer_kwargs
        )
        self.optimizer = optimizer_resolver.make(
            optimizer, optimizer_kwargs, params=self.model.parameters()
        )

    @property
    @abstractmethod
    def model_cls(self) -> Type[DeepBlockerModel]:
        pass

    @abstractmethod
    def create_dataloader(
        self, features: FeatureType, batch_size: int
    ) -> DataLoader[FeatureType]:
        pass

    @abstractmethod
    def run_training_loop(self, train_dataloader: DataLoader, num_epochs: int):
        pass

    def train(
        self,
        features: FeatureType,
        num_epochs: int,
        batch_size: int,
    ) -> DeepBlockerModel:
        self.device = resolve_device()
        self.model.to(self.device)

        train_dataloader = self.create_dataloader(
            features=features, batch_size=batch_size
        )

        self.model.train()
        self.run_training_loop(train_dataloader=train_dataloader, num_epochs=num_epochs)
        self.model.eval()
        return self.model

    def save_model(
        self, output_file_name: Union[str, os.PathLike, BinaryIO, IO[bytes]], **kwargs
    ):
        torch.save(self.model.state_dict(), output_file_name, **kwargs)

    def load_model(
        self, input_file_name: Union[str, os.PathLike, BinaryIO, IO[bytes]], **kwargs
    ):
        self.model = self.model_cls(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name, **kwargs))
        self.model.eval()


class AutoEncoderDeepBlockerModelTrainer(DeepBlockerModelTrainer):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Tuple[int, int],
        learning_rate: float,
        loss_function: _Loss = None,
        optimizer: HintOrType[Optimizer] = None,
        optimizer_kwargs: OptionalKwargs = None,
    ):
        loss_function = nn.MSELoss() if loss_function is None else loss_function
        optimizer = "adam" if optimizer is None else optimizer
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            learning_rate=learning_rate,
            loss_function=loss_function,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

    def create_dataloader(
        self, features: torch.Tensor, batch_size: int
    ) -> DataLoader[torch.Tensor]:
        return DataLoader(
            dataset=FeatureDataset(features), batch_size=batch_size, shuffle=True
        )

    @property
    def model_cls(self) -> Type[DeepBlockerModel]:
        return AutoEncoderDeepBlockerModel

    def run_training_loop(self, train_dataloader: DataLoader, num_epochs: int):
        assert self.loss_function is not None
        for _ in range(num_epochs):
            train_loss = 0
            for _, data in enumerate(train_dataloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, data)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

    def train(
        self,
        features: FeatureType,
        num_epochs: int,
        batch_size: int,
    ) -> DeepBlockerModel:
        return super().train(
            features=features,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )


class CTTDeepBlockerModelTrainer(DeepBlockerModelTrainer):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Tuple[int, int],
        learning_rate: float,
        loss_function: _Loss = None,
        optimizer: HintOrType[Optimizer] = None,
        optimizer_kwargs: OptionalKwargs = None,
    ):
        loss_function = nn.BCELoss() if loss_function is None else loss_function
        optimizer = "adam" if optimizer is None else optimizer
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            learning_rate=learning_rate,
            loss_function=loss_function,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

    def create_dataloader(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int
    ) -> DataLoader:
        left, right, label = features
        return DataLoader(
            dataset=TripletDataset(left, right, label),
            batch_size=batch_size,
            shuffle=True,
        )

    @property
    def model_cls(self) -> Type[DeepBlockerModel]:
        return CTTDeepBlockerModel

    def run_training_loop(self, train_dataloader: DataLoader, num_epochs: int):
        assert self.loss_function is not None
        for _ in range(num_epochs):
            train_loss = 0
            for _, (left, right, label) in enumerate(train_dataloader):
                left = left.to(self.device)
                right = right.to(self.device)
                label = label.unsqueeze(-1)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(left, right)
                loss = self.loss_function(output, label)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

    def train(
        self,
        features: FeatureType,
        num_epochs: int,
        batch_size: int,
    ) -> DeepBlockerModel:
        return super().train(
            features=features,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
