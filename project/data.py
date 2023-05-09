import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
torch.cuda_is_available()
class SquadData(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, batch_size=16, num_workers=4):
        super().__init__()
        self.num_workers = num_workers
        self.val_data = None
        self.train_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prepare_data()

    def prepare_data(self):
        datasets = load_dataset(self.dataset_name)
        self.train_data = datasets["train"]
        self.val_data = datasets["validation"]

    def setup(self, stage=None):
        self.train_dataset = self.train_data.map(self.prepare_examples,
                                                 batched=True,
                                                 remove_columns=self.train_data.column_names)
        self.val_dataset = self.val_data.map(self.prepare_examples,
                                             batched=True,
                                             remove_columns=self.val_data.column_names)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate)

    def custom_collate(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        return batch

    def prepare_examples(self, examples):
        max_length = 384
        doc_stride = 128
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized = self.tokenizer(examples["question" if pad_on_right else "context"],
                                   examples["context" if pad_on_right else "question"],
                                   truncation="only_second" if pad_on_right else "only_first",
                                   max_length=max_length,
                                   stride=doc_stride, return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding="max_length")
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)

            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized["end_positions"].append(token_end_index + 1)

        return tokenized
