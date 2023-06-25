#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from typing import Optional

import torch

from data import BrainDataset
from litsloader import LiTSDataset

class DataDescriptor:

    def __init__(self, n_workers=2, fold=1, batch_size=32, **kwargs):

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataset_cache = {}
        self.fold = fold

    def get_dataset(self, split: str, fold: int):
        raise NotImplemented("get_dataset needs to be overridden in a subclass.")

    def get_dataset_(self, split: str, cache=True, force=False):
        if split not in self.dataset_cache or force:
            dataset = self.get_dataset(split, fold=self.fold)
            if cache:
                self.dataset_cache[split] = dataset
            return dataset
        else:
            return self.dataset_cache[split]

    def get_dataloader(self, split: str):
        dataset = self.get_dataset_(split, cache=True)

        shuffle = True if split == "train" else False
        drop_last = False if len(dataset) < self.batch_size else True

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last)

        return dataloader


class LiverAEDataDescriptor:

    def __init__(self, n_workers=2, fold=1, batch_size=32, **kwargs):

        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataset_cache = {}
        self.fold = fold

    def get_dataset_(self, split: str, cache=True, force=False):
        if split not in self.dataset_cache or force:
            dataset = self.get_dataset(split, fold=self.fold)
            if cache:
                self.dataset_cache[split] = dataset
            return dataset
        else:
            return self.dataset_cache[split]

    def get_dataloader(self, split: str):
        dataset = self.get_dataset_(split, cache=True)

        shuffle = True if split == "train" else False
        drop_last = False if len(dataset) < self.batch_size else True

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last)

        return dataloader

    def get_dataset(self, split: str, fold: int):
        assert split in ["train", "val", "test"]  # "test" should not be used through the DataDescriptor interface in this case.
        if split == 'test':
            test_flag = True
        else:
            test_flag = False
        if split == 'val':
            split = 'test'
        dataset = LiTSDataset(mode=split, fold=fold, test_flag=test_flag)

        return dataset



