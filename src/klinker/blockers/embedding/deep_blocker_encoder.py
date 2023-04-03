# TODO refactor copy-pasted code
# GiG
from class_resolver import ClassResolver

import random
from collections import Counter

import fasttext
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from torchtext.data import get_tokenizer

from klinker.models.deepblocker import AutoEncoderTrainer, CTTModelTrainer
from klinker.blockers.embedding.encoder import FrameEncoder
from klinker.typing import GeneralVector
import pandas as pd
from typing import Tuple
from typing import Tuple


class DeepBlockerEncoder(FrameEncoder):

    def preprocess(self, list_of_tuples):
        raise NotImplementedError

    def get_tuple_embedding(self, list_of_tuples):
        raise NotImplementedError

    def encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        all_merged_text = pd.concat([left["_merged_text"], right["_merged_text"]], ignore_index=True)
        self.preprocess(all_merged_text)
        return self.get_tuple_embedding(left["_merged_text"]), self.get_tuple_embedding(right["_merged_text"])




class AutoEncoderTupleEmbedding(DeepBlockerEncoder):
    def __init__(
        self, input_dimension, hidden_dimensions, fasttext_path#(2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE)
    ):
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions
        self.sif_embedding_model = SIFEmbedding(fasttext_path=fasttext_path)

    def preprocess(self, list_of_tuples):
        print("Training AutoEncoder model")
        self.sif_embedding_model.preprocess(list_of_tuples)
        embedding_matrix = self.sif_embedding_model.get_tuple_embedding(list_of_tuples)
        trainer = AutoEncoderTrainer(self.input_dimension, self.hidden_dimensions)
        self.autoencoder_model = trainer.train(
            embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE
        )

    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(
            self.sif_embedding_model.get_tuple_embedding(list_of_tuples)
        ).float()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)

# This function is used by both CTT and Hybrid  - so it is put outside of any class
# It takes a list of tuple strings and outputs three lists (T, T', L)
# t_i \in T and t'_i \in T' are (potentially perturbed) tuples
# and l_i is a label denoting whether they are duplicates or not
# for each tuple t in list_of_tuples,
# we generate synth_tuples_per_tuple positive tuple pairs
# and synth_tuples_per_tuple * pos_to_neg_ratio negative tuple pairs
def generate_synthetic_training_data(
    list_of_tuples, synth_tuples_per_tuple=5, pos_to_neg_ratio=1, max_perturbation=0.4, random_seed=None
):
    num_positives_per_tuple = synth_tuples_per_tuple
    num_negatives_per_tuple = synth_tuples_per_tuple * pos_to_neg_ratio
    num_tuples = len(list_of_tuples)
    total_number_of_elems = num_tuples * (
        num_positives_per_tuple + num_negatives_per_tuple
    )

    # We create three lists containing T, T' and L respectively
    # We use the following format: first num_tuples * num_positives_per_tuple correspond to T
    # and the remaining correspond to T'
    left_tuple_list = [None for _ in range(total_number_of_elems)]
    right_tuple_list = [None for _ in range(total_number_of_elems)]
    label_list = [0 for _ in range(total_number_of_elems)]

    random.seed(random_seed)

    tokenizer = get_tokenizer("basic_english")
    for index in range(len(list_of_tuples)):
        tokenized_tuple = tokenizer(list_of_tuples[index])
        max_tokens_to_remove = int(len(tokenized_tuple) * max_perturbation)

        training_data_index = index * (
            num_positives_per_tuple + num_negatives_per_tuple
        )

        # Create num_positives_per_tuple tuple pairs with positive label
        for temp_index in range(num_positives_per_tuple):
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

        for temp_index in range(num_negatives_per_tuple):
            left_tuple_list[training_data_index] = list_of_tuples[index]
            right_tuple_list[training_data_index] = random.choice(list_of_tuples)
            label_list[training_data_index] = 0
            training_data_index += 1
    return left_tuple_list, right_tuple_list, label_list


class CTTTupleEmbedding(DeepBlockerEncoder):
    def __init__(
        self,
        input_dimension,
        hidden_dimensions, #(2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE)
        fasttext_path,
        synth_tuples_per_tuple=5,
        pos_to_neg_ratio=1,
        max_perturbation=0.4,
    ):
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation

        # By default, CTT uses SIF as the aggregator
        self.sif_embedding_model = SIFEmbedding(fasttext_path=fasttext_path)

    def preprocess(self, list_of_tuples):
        print("Training CTT model")
        self.sif_embedding_model.preprocess(list_of_tuples)

        (
            left_tuple_list,
            right_tuple_list,
            label_list,
        ) = generate_synthetic_training_data(
            list_of_tuples,
            self.synth_tuples_per_tuple,
            self.pos_to_neg_ratio,
            self.max_perturbation,
        )

        self.left_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(
            left_tuple_list
        )
        self.right_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(
            right_tuple_list
        )
        self.label_list = label_list

        trainer = CTTModelTrainer(self.input_dimension, self.hidden_dimensions)
        self.ctt_model = trainer.train(
            self.left_embedding_matrix,
            self.right_embedding_matrix,
            self.label_list,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(
            self.sif_embedding_model.get_tuple_embedding(list_of_tuples)
        ).float()
        return embedding_matrix


class HybridTupleEmbedding(DeepBlockerEncoder):
    def __init__(
        self,
        hidden_dimensions, #(2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE),
        fasttext_path,
        input_dimension: int = 300,
        synth_tuples_per_tuple=5,
        pos_to_neg_ratio=1,
        max_perturbation=0.4,
    ):
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation

        # Hybrid uses autoencoder instead of SIF aggregator
        self.autoencoder_embedding_model = AutoEncoderTupleEmbedding(input_dimension=input_dimension, hidden_dimensions=hidden_dimensions, fasttext_path=fasttext_path)

    # This function is used as a preprocessing step
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, list_of_tuples):
        print("Training CTT model")
        self.autoencoder_embedding_model.preprocess(list_of_tuples)

        (
            left_tuple_list,
            right_tuple_list,
            label_list,
        ) = generate_synthetic_training_data(
            list_of_tuples,
            self.synth_tuples_per_tuple,
            self.pos_to_neg_ratio,
            self.max_perturbation,
        )

        self.left_embedding_matrix = (
            self.autoencoder_embedding_model.get_tuple_embedding(left_tuple_list)
        )
        self.right_embedding_matrix = (
            self.autoencoder_embedding_model.get_tuple_embedding(right_tuple_list)
        )
        self.label_list = label_list

        trainer = CTTModelTrainer(self.input_dimension, self.hidden_dimensions)
        self.ctt_model = trainer.train(
            self.left_embedding_matrix,
            self.right_embedding_matrix,
            self.label_list,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

    # This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(
            self.autoencoder_embedding_model.get_tuple_embedding(list_of_tuples)
        ).float()
        return embedding_matrix

    # This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(
            self.autoencoder_embedding_model.get_tuple_embedding(list_of_words)
        ).float()
        return embedding_matrix

deep_blocker_encoder_resolver = ClassResolver([ AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding], base=DeepBlockerEncoder, default=CTTTupleEmbedding)
