import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("multi_task_learning_with_t5")

import src.config as config
from src.preprocess import preprocess_data

# クラスの違いは2点
# (1)prefixを付与するか否か
# (2)各バッチにtask_typeを含むか否か


class SingleTaskDataset(Dataset):
    def __init__(self, task_type, split, data_limit=None, label_encode=False):
        super(SingleTaskDataset, self).__init__()
        self.task_type = task_type
        self.split = split
        self.label_encode = label_encode

        if task_type == "MARC":
            data_path = os.path.join(config.MARC_DIR, f"{split}.tsv")
            self.data = pd.read_csv(data_path, sep="\t")
            if self.label_encode:
                label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
                self.data["label"] = self.data["label"].replace(label_mapping)

        elif task_type == "AMCD":
            data_path = os.path.join(config.AMCD_DIR, f"{split}.tsv")
            self.data = pd.read_csv(data_path, sep="\t")

        if data_limit:
            try:
                if task_type == "MARC":
                    self.data, _ = train_test_split(
                        self.data, train_size=data_limit, stratify=self.data["label"]
                    )
                    self.data.reset_index(drop=True, inplace=True)  # インデックスをリセット
                elif task_type == "AMCD":
                    self.data, _ = train_test_split(
                        self.data,
                        train_size=data_limit,
                        stratify=self.data["is_counterfactual"],
                    )
                    self.data.reset_index(drop=True, inplace=True)
            except ValueError:
                self.data = self.data.sample(n=data_limit, random_state=12)
                self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task_type == "MARC":
            input_text = self.data.loc[idx, "text"]
            if self.label_encode:
                target_text = str(self.data.loc[idx, "label"])
            else:
                target_text = self.data.loc[idx, "label"]
        elif self.task_type == "AMCD":
            input_text = self.data.loc[idx, "sentence"]
            target_text = str(self.data.loc[idx, "is_counterfactual"])

        input_text = preprocess_data(input_text)
        return {"input_text": input_text, "target_text": target_text}


class MultiTaskDataset(Dataset):
    def __init__(self, task_type, split, data_limit=None, label_encode=False):
        super(MultiTaskDataset, self).__init__()
        self.task_type = task_type
        self.split = split
        self.label_encode = label_encode

        if task_type == "MARC":
            data_path = os.path.join(config.MARC_DIR, f"{split}.tsv")
            self.data = pd.read_csv(data_path, sep="\t")
            self.prefix = "marc: "
            if self.label_encode:
                label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
                self.data["label"] = self.data["label"].replace(label_mapping)
        elif task_type == "AMCD":
            data_path = os.path.join(config.AMCD_DIR, f"{split}.tsv")
            self.data = pd.read_csv(data_path, sep="\t")
            self.prefix = "amcd: "

        if data_limit:
            try:
                if task_type == "MARC":
                    self.data, _ = train_test_split(
                        self.data, train_size=data_limit, stratify=self.data["label"]
                    )
                    self.data.reset_index(drop=True, inplace=True)  # インデックスをリセット
                elif task_type == "AMCD":
                    self.data, _ = train_test_split(
                        self.data,
                        train_size=data_limit,
                        stratify=self.data["is_counterfactual"],
                    )
                    self.data.reset_index(drop=True, inplace=True)
            except ValueError:
                self.data = self.data.sample(n=data_limit, random_state=12)
                self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task_type == "MARC":
            input_text = self.prefix + self.data.loc[idx, "text"]
            if self.label_encode:
                target_text = str(self.data.loc[idx, "label"])
            else:
                target_text = self.data.loc[idx, "label"]
            task_type = "MARC"
        elif self.task_type == "AMCD":
            input_text = self.prefix + self.data.loc[idx, "sentence"]
            target_text = str(self.data.loc[idx, "is_counterfactual"])
            task_type = "AMCD"

        input_text = preprocess_data(input_text)

        return {"input_text": input_text, "target_text": target_text, "task_type": task_type}
