# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import datetime
import json
import random

import torch
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

QUERIES = [
    "How might the {coin} price change?",
    "What kind of changes could occur in the future for the {coin} price?",
    "How could the {coin} price trend evolve?",
    "Will {coin} rise or fall? What's your perspective?",
    "Will {coin} rise or fall? What's your guess?",
    "Will {coin} continue to rise or fall, or will there be a correction?",
    "What developments have occurred in {coin}'s recent price?",
    "What's the situation with {coin}'s recent price?",
    "Which direction will {coin}'s price move in?",
    "What kind of changes can be expected in {coin}'s price?",
    "What will be the trend of {coin}'s price?",
    "How will {coin}'s price change?",
    "Will the momentum of {coin}'s price trend continue?",
    "Has the price of {coin} recently increased or decreased?",
    "What's the recent trend of {coin}?",
    "What are your thoughts on the future direction of {coin}'s price?",
    "Do you think {coin} will rise or fall?",
    "Do you think {coin} will continue to rise or experience a decline?",
    "Do you believe {coin} will keep rising or start declining?",
    "What kind of fluctuations do you expect in {coin}'s price in the coming days?",
    "What changes do you anticipate in {coin}'s price in the coming days?",
    "Do you have any predictions or suggestions regarding the trend of {coin}'s price?",
    "What is your viewpoint on the recent trend of {coin}?",
    "Can you provide analysis and predictions regarding {coin}'s price movements?",
    "Can you offer any predictions about {coin}'s price?",
    "What are your predictions for {coin}'s price in the next few days?",
    "What are your expectations for {coin}'s price in the next few days?",
    "What are your thoughts on the future of {coin}'s price in the coming days?",
    "What are your insights on the recent fluctuations in {coin}'s price?",
    "What are your thoughts on the prospects of {coin}'s price?",
    "What is your perspective on the future development trends of {coin}'s price?",
    "What are your thoughts on the short-term trend of {coin}'s price?",
    "What is your viewpoint on the recent performance of {coin}'s price?",
    "What do you think about the recent trend of {coin}'s price?",
    "Can you provide some analysis on the recent price of {coin}?",
    "Can you predict the future trend of {coin}'s price?",
    "Do you think {coin}'s price will continue to rise or might it experience a decline?",
    "Do you think {coin} will continue to rise or fall, or could there be a correction?",
    "What has been the recent trend in {coin}'s price?",
    "How has the price of {coin} been recently?",
    "What has been the recent price movement of {coin}?",
    "What changes have occurred in the recent {coin} price?",
    "What has been the recent trend in the price of {coin}?",
    "What can be expected for the {coin} price in the coming days?",
    "How might the price of {coin} change in the coming days?",
    "What are your predictions for the price of {coin} in the near future?",
    "Please analyze the trend of {coin}'s price.",
    "Please provide a detailed analysis of {coin}'s price.",
    "Please analyze the recent price of {coin}.",
    "How is the recent {coin} market performing?",
]


class InstructionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        tokenizer,
        partition="train",
    ) -> None:
        if partition == "train":
            path = dataset_config.train_split
        else:
            path = dataset_config.test_split

        with open(path) as f:
            self.examples = json.load(f)
            self.examples = [
                x for x in self.examples if len(str(x)) < dataset_config.input_length
            ]

        self.max_length = dataset_config.input_length
        self.tokenizer = tokenizer

        self.instruction_template = "### Human:"
        self.response_template = "### Assistant:"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        def _ts_to_str(ts):
            return datetime.datetime.fromtimestamp(ts).date().isoformat()

        example = self.examples[index]
        date = _ts_to_str(example["created_at"])
        coin = example["pair"].split("/")[0]
        question = random.choice(QUERIES).format(coin=coin)
        answer = example["text"]
        columns = [
            "Date",
            "Opening",
            "Highest",
            "Lowest",
            "Closing",
            "Volume",
            "Closing EMA(20)",
            "Closing SMA(50)",
            "Closing SMA(200)",
            "Closing RSI",
        ]

        prices = "\n".join(
            ", ".join(
                f"{k}: {v if k != 'Date' else _ts_to_str(v)}"
                for k, v in zip(columns, values)
            )
            for values in example["ohlcv"]
        )

        prompt = f"""{self.instruction_template}
Today is {date}. Below is the recent price info of {coin}. As a senior investment analyst, please answer this question: {question}

{prices}

"""

        response = f"""{self.response_template}
{answer}"""

        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        example = prompt + response

        prompt = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.int64,
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)

        padding = self.max_length - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_length]

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
