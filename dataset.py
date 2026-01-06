from typing import Callable, List, Tuple, TypedDict
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from urllib.request import urlretrieve
import os
import zipfile
import librosa
import random
import shutil

DATA_PATH: str = "data"
CACHE_PATH: str = "cache"


class DatasetInfo:
    prepare_data: Callable[[], str]
    dataset_name: str
    words: list[str]

    def __init__(
        self, prepare_data: Callable[[], str], dataset_name: str, words: list[str]
    ):
        self.prepare_data = prepare_data
        self.dataset_name = dataset_name
        self.words = words


def extract_word_speaker_index(file_path: str) -> tuple[str, str, str]:
    word: str = os.path.dirname(file_path)
    [speaker, idx] = (
        os.path.basename(file_path)
        .replace("_nohash", "")
        .replace(".wav", "")
        .split("_")
    )
    return (word, speaker, idx)


def extract_mfcc(wav_path: str, sr: int = 16000, n_mfcc: int = 40) -> torch.Tensor:
    """Extract MFCC features from an audio file.

    Returns a float tensor of shape (time, n_mfcc).
    """
    y, _loaded_sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return torch.tensor(mfcc.T, dtype=torch.float32)


def cache_mfccs(dataset_name: str, sr: int = 16000, n_mfcc: int = 40):
    dataset_path: str = os.path.join(DATA_PATH, dataset_name)
    words: list[str] = [
        name
        for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    ]
    wav_paths: list[str] = []
    cached_paths: list[str] = []
    for word in words:
        file_paths: list[str] = [
            file_path
            for file_path in os.listdir(os.path.join(dataset_path, word))
            if os.path.isfile(os.path.join(dataset_path, word, file_path))
            and file_path.endswith(".wav")
        ]
        for file_path in file_paths:
            wav_path: str = os.path.join(dataset_path, word, file_path)
            cached_path: str = (
                f"{CACHE_PATH}/{dataset_name}/{word}/{file_path.replace('wav', 'pt')}"
            )
            wav_paths.append(wav_path)
            cached_paths.append(cached_path)
    total_len: int = len(wav_paths)
    total_omitted: int = 0
    for index, (wav_path, cached_path) in enumerate(zip(wav_paths, cached_paths)):
        if os.path.exists(cached_path):
            total_omitted += 1
            continue
        else:
            mfcc = extract_mfcc(wav_path, sr, n_mfcc)
            os.makedirs(os.path.dirname(cached_path), exist_ok=True)
            torch.save(mfcc.cpu(), cached_path)
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{total_len} cached [{total_omitted} omitted]")


def extract_or_cache_mfcc(
    dataset_name: str, word: str, file_name: str, sr: int = 16000, n_mfcc: int = 40
) -> torch.Tensor:
    """Extract MFCC features from an audio file.

    Returns a float tensor of shape (time, n_mfcc).
    """
    wav_path: str = f"{DATA_PATH}/{dataset_name}/{word}/{file_name}"
    cached_path: str = (
        f"{CACHE_PATH}/{dataset_name}/{word}/{file_name.replace('wav', 'pt')}"
    )
    os.makedirs(CACHE_PATH, exist_ok=True)
    in_cache: bool = os.path.exists(cached_path)
    if not in_cache:
        mfcc = extract_mfcc(wav_path, sr, n_mfcc)
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        torch.save(mfcc.cpu(), cached_path)
    else:
        mfcc = torch.load(cached_path, map_location="cpu")
    return mfcc


class PKBatchSampler(Sampler[list[int]]):
    """Samples batches with P classes and K samples/class (batch size = P*K)."""

    def __init__(
        self,
        label_to_indices: dict[int, list[int]],
        P: int,
        K: int,
        steps_per_epoch: int,
        seed: int = 0,
    ):
        if P <= 0 or K <= 0:
            raise ValueError("P and K must be positive")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.label_to_indices = label_to_indices
        self.labels = sorted(self.label_to_indices.keys())
        if len(self.labels) < P:
            raise ValueError(f"Need at least P={P} classes, found {len(self.labels)}")

        self.P = P
        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen_labels = rng.sample(self.labels, self.P)
            batch_indices: list[int] = []
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.K:
                    batch_indices.extend(rng.sample(indices, self.K))
                else:
                    batch_indices.extend(rng.choices(indices, k=self.K))
            rng.shuffle(batch_indices)
            yield batch_indices


class DataDataset(Dataset):
    def __init__(self, dataset_path: str, words: List[str], n_mfcc: int, sr: int):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.label_to_idx: dict[str, int] = {w: i for i, w in enumerate(words)}
        self.items: List[tuple[str, str]] = []
        self.dataset_path = dataset_path
        count: int = 0
        for word in words:
            for item_name in os.listdir(f"{dataset_path}/{word}"):
                if item_name.endswith(".wav"):
                    file_path: str = f"{dataset_path}/{word}/{item_name}"
                    self.items.append((file_path, word))
                    count += 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[index]
        word: str = os.path.basename(os.path.dirname(path))
        file_name: str = os.path.basename(path)
        return extract_or_cache_mfcc(
            os.path.basename(self.dataset_path),
            word,
            file_name,
            n_mfcc=self.n_mfcc,
            sr=self.sr,
        ), self.label_to_idx[label]


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        n_mfcc: int,
        sr: int,
        p: int,
        k: int,
        steps_per_epoch: int,
        val_steps_per_epoch: int,
        dataset_info: DatasetInfo,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.p = p
        self.k = k
        self.steps_per_epoch = steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.dataset_info = dataset_info
        self.dataset_path: str = f"{DATA_PATH}/{self.dataset_info.dataset_name}"

    def prepare_data(self) -> None:
        print(f"Preparing dataset: {self.dataset_info.dataset_name}")
        if os.path.exists(self.dataset_path):
            print(
                f"{self.dataset_path} already exists, using it. If you want to reload, remove the directory"
            )
        else:
            dataset_import_path: str = self.dataset_info.prepare_data()
            shutil.copytree(dataset_import_path, self.dataset_path)

    def setup(self, stage: str) -> None:
        seen_ds = DataDataset(
            self.dataset_path, self.dataset_info.words, n_mfcc=self.n_mfcc, sr=self.sr
        )
        self.train_ds, self.val_ds, self.test_ds = random_split(
            seen_ds, [0.8, 0.1, 0.1]
        )

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, int]]):
        xs, ys = zip(*batch)  # xs: tuple[(T_i, n_mfcc)], ys: tuple[int]
        lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.int64)
        x_padded = pad_sequence(list(xs), batch_first=True)  # (B, T_max, n_mfcc)
        y = torch.tensor(list(ys), dtype=torch.long)
        return x_padded, lengths, y

    def _label_to_subset_indices(self, subset) -> dict[int, list[int]]:
        """
        Build label->indices mapping where indices are positions within the Subset
        (i.e., what DataLoader will index into).
        """
        base_ds = subset.dataset  # DataDataset
        mapping: dict[int, list[int]] = {}
        for subset_pos, base_idx in enumerate(subset.indices):
            _path, word = base_ds.items[base_idx]
            label = base_ds.label_to_idx[word]
            mapping.setdefault(label, []).append(subset_pos)
        return mapping

    def train_dataloader(self):
        label_to_indices = self._label_to_subset_indices(self.train_ds)
        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            batch_sampler=PKBatchSampler(
                label_to_indices, self.p, self.k, self.steps_per_epoch
            ),
        )

    def val_dataloader(self):
        label_to_indices = self._label_to_subset_indices(self.val_ds)
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            batch_sampler=PKBatchSampler(
                label_to_indices, self.p, self.k, self.val_steps_per_epoch
            ),
        )

    def test_dataloader(self):
        label_to_indices = self._label_to_subset_indices(self.test_ds)
        return DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            batch_sampler=PKBatchSampler(
                label_to_indices, self.p, self.k, self.steps_per_epoch
            ),
        )
