import os
import random
from dataclasses import dataclass
from typing import Final

import librosa
import torch
from torch.utils.data import Dataset, Sampler


DEFAULT_AUDIO_DIR: Final[str] = "data/speech_commands_v0.02/"

WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

UNKNOWN_WORDS_V1 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
]

UNKNOWN_WORDS_V2 = UNKNOWN_WORDS_V1 + [
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]

SILENCE = "_silence_"  # background noise
LABELS_V1 = WORDS + UNKNOWN_WORDS_V1 + [SILENCE]
LABELS_V2 = WORDS + UNKNOWN_WORDS_V2 + [SILENCE]

# Convenience variants for training/eval when you want to ignore silence.
LABELS_V1_NO_SILENCE = [w for w in LABELS_V1 if w != SILENCE]
LABELS_V2_NO_SILENCE = [w for w in LABELS_V2 if w != SILENCE]


def extract_mfcc(wav_path: str, sr: int = 16000, n_mfcc: int = 40) -> torch.Tensor:
    """Extract MFCC features from an audio file.

    Returns a float tensor of shape (time, n_mfcc).
    """
    y, _loaded_sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return torch.tensor(mfcc.T, dtype=torch.float32)


def list_words(audio_dir: str = DEFAULT_AUDIO_DIR) -> list[str]:
    """List label folders under the Speech Commands dataset root."""
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    words: list[str] = []
    for name in sorted(os.listdir(audio_dir)):
        full = os.path.join(audio_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("_"):
            continue
        words.append(name)
    return words


def _normalize_relpath(path: str) -> str:
    return path.replace("\\", "/")


def read_split_list(list_path: str) -> set[str]:
    """Read Speech Commands split file (paths relative to dataset root)."""
    relpaths: set[str] = set()
    if not os.path.isfile(list_path):
        return relpaths
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            relpaths.add(_normalize_relpath(rel))
    return relpaths


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor  # (B, T, F)
    lengths: torch.Tensor  # (B,)
    labels: torch.Tensor  # (B,)


class SpeechCommandsDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        audio_dir: str = DEFAULT_AUDIO_DIR,
        words: list[str] | None = None,
        sr: int = 16000,
        n_mfcc: int = 40,
        unique_speakers: bool = False,
        max_items_per_word: int | None = None,
        include_relpaths: set[str] | None = None,
        exclude_relpaths: set[str] | None = None,
    ):
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mfcc = n_mfcc

        if words is None:
            words = list_words(audio_dir)
        else:
            # The official Speech Commands label lists include a `_silence_` class,
            # but the local folder-based dataset loader doesn't use it.
            words = [w for w in words if w != SILENCE]
        if not words:
            raise ValueError("No word folders found")

        self.words = list(words)
        self.word_to_label = {w: i for i, w in enumerate(self.words)}

        items: list[tuple[str, int]] = []
        for word in self.words:
            word_dir = os.path.join(audio_dir, word)
            if not os.path.isdir(word_dir):
                continue

            wavs = [
                os.path.join(word_dir, fn)
                for fn in os.listdir(word_dir)
                if fn.endswith(".wav")
            ]
            wavs.sort()

            if include_relpaths is not None or exclude_relpaths is not None:
                filtered: list[str] = []
                for path in wavs:
                    rel = _normalize_relpath(os.path.relpath(path, audio_dir))
                    if include_relpaths is not None and rel not in include_relpaths:
                        continue
                    if exclude_relpaths is not None and rel in exclude_relpaths:
                        continue
                    filtered.append(path)
                wavs = filtered

            if unique_speakers:
                speakers: set[str] = set()
                filtered = []
                for path in wavs:
                    speaker_id = os.path.basename(path).split("_nohash_")[0]
                    if speaker_id in speakers:
                        continue
                    speakers.add(speaker_id)
                    filtered.append(path)
                wavs = filtered

            if max_items_per_word is not None:
                wavs = wavs[:max_items_per_word]

            label = self.word_to_label[word]
            items.extend((p, label) for p in wavs)

        if not items:
            raise ValueError(f"No .wav files found under {audio_dir}")

        self.items = items

        self.label_to_indices: dict[int, list[int]] = {}
        for idx, (_, label) in enumerate(self.items):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        wav_path, label = self.items[index]
        x = extract_mfcc(wav_path, sr=self.sr, n_mfcc=self.n_mfcc)
        return x, label


def collate_pad(batch: list[tuple[torch.Tensor, int]]) -> Batch:
    xs, labels = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(xs[0].shape[1])
    padded = torch.zeros((len(xs), max_len, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0], :] = x
    return Batch(
        x=padded,
        lengths=lengths,
        labels=torch.tensor(labels, dtype=torch.long),
    )


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

        self.label_to_indices = {
            int(label): list(indices) for label, indices in label_to_indices.items()
        }
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
