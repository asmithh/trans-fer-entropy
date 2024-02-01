import os
from transformers import pipeline, Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bart.modeling_bart import BartForSequenceClassification
from torch.utils.data import DataLoader
from torch import Tensor
import torch
from typing import Union, Tuple, Dict, List
from pandas import DataFrame
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Articles collected from MediaCloud are often not all relevant to the topic of trans discourses, so this script uses a 
pretrained LLM to filter out the irrelevant articles. 

- Sagar Kumar, 2024
"""

MODEL = "facebook/bart-large-mnli"


class ZeroShotOutput:

    def __init__(self,
                 indices: List[int],
                 labels: List[str],
                 sequences: List[str],
                 scores: List[Dict[str, float]]) -> None:

        """
        Output object for the zero shot classifier. Requires only the possible labels that are being classified.

        :param indices: indices of each item
        :param labels: set of labels to classify from
        :param sequences: sequences output from pipeline
        :param scores: scores output from pipeline
        """

        assert len(indices) == len(sequences) == len(scores), "Mismatch between index, sequence, and score lengths."

        self.data = tuple(zip(indices, sequences, scores))
        self.labels = labels


class ZeroShot:

    def __init__(self,
                 model: str = MODEL,
                 cuda: Union[bool, str] = True
                 ) -> None:

        """
        Wrapper for the huggingface zeroshot pipeline with additional details regarding dataset, batching, labels, etc.

        :param model: Zero Shot-optimized model chosen from huggingface library
        :param cuda: run on CUDA if True, CPU if False. Alternatively, specify GPU or CPU.
        """

        if isinstance(cuda,bool):
            if cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cuda)

        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def classify(self,
                 items: Union[np.ndarray[str], Tuple[str]],
                 labels: List[str],
                 max_length: int = 258) -> List[Dict[str, float]]:

        """
        Classifies a list/array of instances with fixed possible labels.

        :param items: List of sequences (headlines) to classify
        :param labels: List of possible labels
        :param max_length: truncates any sequences with more than this number of characters (to reduce complexity)
        :return: list of dicts for each sequence with corresponding sequence, list of labels, and list of scores
        {'labels', 'scores', 'sequence'}
        """

        # TODO: max length will be implemented in the tokenizer or dataloader

        predictions = list()

        for item in items:
            premise = item

            scores: Dict[str, float] = {lbl: 0 for lbl in labels}

            for lbl in labels:
                hypothesis:str = f"This example is {lbl}."

                x: Tensor = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first',
                                                  max_length=max_length)

                logits: Tensor = self.model(x.to(self.device))[0]

                non_neutral_logits: Tensor = logits[:, [0, 2]]  # Throwing out the neutral dimension

                probs: Tensor = non_neutral_logits.softmax(dim=1)

                prob_true = probs[:, 1].item()

                scores[lbl] = prob_true

            predictions.append(scores)

        return predictions

    def batch_classify(self,
                       labels: List[str],
                       data: DataFrame,
                       col_name: str = 'title',
                       batch_size: int = 8) -> ZeroShotOutput:

        """
        Run the Zero-Shot Pipeline in batches over a large collection of items in a csv file.

        :param labels: list of possible classification labels
        :param data: data in the form of a Pandas DataFrame
        :param col_name: name of the CSV column containing relevant sequence
        :param batch_size: number of sequences to put through the pipeline at once
        :return: ZeroShotOutput object with sequences and scores all appended into one long list
        """

        scores = list()

        indices = data.index
        sequences = data[col_name]

        splits = DataLoader(sequences, batch_size=batch_size)
        iter_dl = tqdm(splits, desc='Classifying Batch')

        for x in iter_dl:
            out: List[Dict[str, float]] = self.classify(x, labels)
            scores.extend(out)

        # print(len(out_sequences), len(out_scores), len(indices))

        output = ZeroShotOutput(labels=labels,
                                indices=indices,
                                sequences=sequences,
                                scores=scores)

        return output

    def filter(self,
               data: DataFrame,
               target_label: str,
               labels: List[str],
               threshold: float,
               col_name: str = 'title',
               batch_size: int = 8
               ) -> DataFrame:

        """
        Run classification and filter based on target label threshold score
        :param data: data in the form of a Pandas DataFrame
        :param target_label: Label being filtered for
        :param labels: possible classification labels
        :param threshold: minimum score to accept headline
        :param col_name: name of the CSV column containing relevant sequence
        :param batch_size: number of sequences to put through the pipeline at once
        :return:
        """

        assert target_label in labels, "target label must in the possible labels"

        preds: ZeroShotOutput = self.batch_classify(
            labels=labels,
            col_name=col_name,
            batch_size=batch_size,
            data=data
        )

        pos_indices = list()

        for idx, seq, score in tqdm(preds.data, desc="Filtering"):
            if score[target_label] > threshold:
                pos_indices.append(idx)

        og_df = data
        out_df = og_df.iloc[pos_indices]

        return out_df




















