import os
from transformers import pipeline, Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers.models.bart.modeling_bart import BartForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch
from typing import Union, Tuple, Dict, List
from pandas import DataFrame
import pandas as pd
import numpy as np
from tqdm import tqdm
from filter_utils import special_tokens_dict, Collator
import wandb


class CustomDataset(Dataset):

    def __init__(self,
                 datapath: str,
                 text_columns: list,
                 encode_labels: bool = True) -> None:
        """
        Custom implementation of the pytorch Dataset class to allow us to use our own pandas dataframes as data.
        :param datapath: filepath to data as comma-separated CSV with label column title "label"
        :param text_columns: list of text columns concatenated to use as model input
        """

        self.data = pd.read_csv(datapath)
        self.cols = text_columns
        self.encode_labels = encode_labels

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[str, str]:
        texts = [str(self.data[col][idx]) for col in self.cols]
        out_str = " ".join(texts)

        if self.encode_labels:
            label_int = self.data['label'][idx]

            if label_int:
                label = 'transgender'
            else:
                label = 'not transgender'

        else:
            label = self.data['label'][idx]

        return out_str, label


class FewShotFineTuner:

    def __init__(self,
                 train_path: str,
                 text_columns: List[str],
                 pretrained_model: str,
                 model_name: str,
                 cuda: Union[bool, str] = True,
                 encode_labels: bool = True
                 ) -> None:

        """
        Code to Fine Tune pretrained model using custom dataset.
        :param train_path: filepath to data as comma-separated CSV with label column title "label"
        :param text_columns: list of text columns concatenated to use as model input
        :param pretrained_model: pretrained Huggingface model name
        :param cuda: run on CUDA if True, CPU if False. Alternatively, specify GPU or CPU.
        :param model_name: output name for model
        :param encode_labels: whether to encode labels
        """

        if isinstance(cuda, bool):
            if cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cuda)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, normalization=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

        self.model.to(self.device)

        collate = Collator(self.tokenizer)
        train_data = CustomDataset(datapath=train_path, text_columns=text_columns, encode_labels=encode_labels)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=1,
                                  collate_fn=collate)

        optimizer = AdamW(self.model.parameters(), lr=10e-5)

        wandb.init(project=model_name)
        wandb.watch(self.model, log_freq=1, log="all")

        pbar = tqdm(train_loader, desc='Fine Tuning')

        self.model.train()
        for tup in pbar:
            x, masks, y = tup[0].to(self.device), tup[1].to(self.device), tup[2].to(self.device)
            outputs = self.model(input_ids=x, labels=y, attention_mask=masks)
            loss = outputs.loss
            wandb.log({"loss":loss})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.save(self.model.state_dict(), f"/data_users1/sagar/trans-fer-entropy/models/{model_name}.pth")
        print('Training Complete')
