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
import tempfile

DATA_PATH: str = "data"
CACHE_PATH: str = "cache"


class DatasetInfo:
    prepare_data: Callable[[], str]
    dataset_name: str
    seen_words: list[str]
    unseen_words: list[str]

    def __init__(
        self,
        prepare_data: Callable[[], str],
        dataset_name: str,
        words: list[str],
        unseen_words: list[str],
    ):
        self.prepare_data = prepare_data
        self.dataset_name = dataset_name
        self.seen_words = words
        self.unseen_words = unseen_words

    def dataset_path(self) -> str:
        return os.path.join(DATA_PATH, self.dataset_name)

    def word_path(self, word: str) -> str:
        return os.path.join(self.dataset_path(), word)

    def sample_word(self, word: str, n: int) -> list[str]:
        word_dir: str = self.word_path(word)
        all_files: list[str] = [
            os.path.join(DATA_PATH, self.dataset_name, word, f)
            for f in os.listdir(word_dir)
            if os.path.isfile(os.path.join(word_dir, f)) and f.endswith(".wav")
        ]
        return random.sample(all_files, n)


def extract_dataset_word_filename(file_path: str) -> Tuple[str, str, str]:
    """Deconstruct file path into (dataset, word, filename)"""
    parts = file_path.split(os.sep)
    dataset = parts[-3]
    word = parts[-2]
    filename = parts[-1]
    return (dataset, word, filename)


def draw_embeddings(
    save_path: str,
    dataset_info: DatasetInfo,
    model: L.LightningModule,
    words: list[str],
):
    samples: list[str] = []
    # k_seen_words: int = 5
    # if len(dataset_info.seen_words) < k_seen_words:
    #     k_seen_words = len(dataset_info.seen_words)
    # k_unseen_words: int = 5
    # if len(dataset_info.unseen_words) < k_unseen_words:
    #     k_unseen_words = len(dataset_info.unseen_words)
    # for word in random.sample(dataset_info.seen_words, k=k_seen_words) + random.sample(
    #     dataset_info.unseen_words, k=k_unseen_words
    # ):
    #     samples.extend(dataset_info.sample_word(word, n=10))
    for word in words:
        samples.extend(dataset_info.sample_word(word, n=10))
    embeddings: list[torch.Tensor] = []
    labels: list[str] = []
    for sample in samples:
        dataset, word, filename = extract_dataset_word_filename(sample)
        mfcc = extract_or_cache_mfcc(dataset, word, filename)
        embedding = model(mfcc.unsqueeze(0)).squeeze(0)
        embeddings.append(embedding)
        if word in dataset_info.seen_words:
            labels.append(f"{word}")
        else:
            labels.append(f"{word} (unseen)")
    draw_embedding_map_from_lists(
        embeddings,
        labels,
        show=False,
        save_path=save_path,
    )


def draw_embedding_map_from_lists(
    embeddings: list[torch.Tensor],
    labels: list[str],
    *,
    perplexity: float = 30.0,
    title: str = "t-SNE Embedding Map",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Convenience wrapper around draw_embedding_map().

    Args:
      embeddings: list of 1D tensors (embedding vectors), length N.
      labels: list of string labels, length N.
    """
    if len(embeddings) != len(labels):
        raise ValueError(
            f"draw_embedding_map_from_lists: embeddings and labels must have same length "
            f"({len(embeddings)} vs {len(labels)})."
        )
    if len(embeddings) == 0:
        print("draw_embedding_map_from_lists: empty input, skipping plot.")
        return

    # Stack embeddings into (N, D)
    emb0 = embeddings[0].detach().flatten()
    d = int(emb0.numel())
    for i, e in enumerate(embeddings):
        e_flat = e.detach().flatten()
        if int(e_flat.numel()) != d:
            raise ValueError(
                f"draw_embedding_map_from_lists: embedding at index {i} has dim {int(e_flat.numel())}, expected {d}."
            )

    X = torch.stack([e.detach().flatten() for e in embeddings], dim=0)

    # Map string labels -> int ids (stable order by first appearance)
    label_to_id: dict[str, int] = {}
    labels_to_strings: list[str] = []
    y_ids: list[int] = []
    for s in labels:
        if s not in label_to_id:
            label_to_id[s] = len(labels_to_strings)
            labels_to_strings.append(s)
        y_ids.append(label_to_id[s])

    y = torch.tensor(y_ids, dtype=torch.long)

    # Delegate plotting + perplexity clamping to the existing function
    return draw_embedding_map(
        X,
        y,
        perplexity=perplexity,
        title=title,
        labels_to_strings=labels_to_strings,
        save_path=save_path,
        show=show,
    )


def draw_embedding_map(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    perplexity: float = 30.0,
    title: str = "t-SNE Embedding Map",
    labels_to_strings: list[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Draw 2D embedding map using t-SNE.

    Notes:
      - t-SNE requires 0 < perplexity < n_samples.
      - In headless/non-interactive environments (e.g. Matplotlib backend 'Agg'),
        the plot is saved to disk instead of calling plt.show().

    labels_to_strings:
      Global lookup table: labels_to_strings[label_id] -> human readable name.
      Label ids may be sparse and do not need to start at 0, but must be valid
      indices into labels_to_strings.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    X = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    n_samples = int(X.shape[0])

    # Nothing useful to plot
    if n_samples < 2:
        print("draw_embedding_map: n_samples < 2, skipping plot.")
        return

    # t-SNE constraint: perplexity must be strictly less than n_samples
    safe_perplexity = float(perplexity)
    safe_perplexity = min(safe_perplexity, float(n_samples - 1))
    safe_perplexity = max(1.0, safe_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    embeddings_2d = tsne.fit_transform(X)

    # Build a stable label->color mapping so we can use a discrete colormap with
    # enough unique colors for many classes (tab10 repeats after 10).
    unique_labels = sorted({int(v) for v in y.tolist()})
    label_to_color_idx: dict[int, int] = {lab: i for i, lab in enumerate(unique_labels)}
    color_ids = [label_to_color_idx[int(v)] for v in y.tolist()]
    n_classes = max(1, len(unique_labels))

    # Pick a categorical colormap. Prefer tab10/tab20 when possible; fall back
    # to an HSV wheel for larger numbers of classes.
    cmap_name = "tab10" if n_classes <= 10 else ("tab20" if n_classes <= 20 else "hsv")
    try:
        # Matplotlib 3.5+: resampled() gives an N-sized discrete cmap.
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n_classes)
    except Exception:
        # Older Matplotlib: get_cmap(name, lut)
        cmap = plt.get_cmap(cmap_name, n_classes)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=color_ids,
        cmap=cmap,
        vmin=-0.5,
        vmax=n_classes - 0.5,
        alpha=0.7,
    )
    # Build a legend with consistent colors. If labels_to_strings is provided,
    # use it as a global lookup by label id; otherwise show numeric ids.
    import matplotlib.lines as mlines

    if labels_to_strings is not None and unique_labels:
        min_lab = min(unique_labels)
        max_lab = max(unique_labels)
        if min_lab < 0:
            raise ValueError(
                f"draw_embedding_map: negative label id {min_lab} cannot index labels_to_strings."
            )
        if max_lab >= len(labels_to_strings):
            raise ValueError(
                "draw_embedding_map: label id out of range for labels_to_strings "
                f"(max label={max_lab}, len(labels_to_strings)={len(labels_to_strings)})."
            )

    handles: list[mlines.Line2D] = []
    legend_names: list[str] = []
    for lab in unique_labels:
        idx = label_to_color_idx[lab]
        color = cmap(idx)
        name = labels_to_strings[lab] if labels_to_strings is not None else str(lab)
        handles.append(
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.7,
            )
        )
        legend_names.append(name)

    if handles:
        plt.legend(handles, legend_names, title="Classes")

    plt.title(f"{title} (perplexity={safe_perplexity:g}, n={n_samples})")

    backend = matplotlib.get_backend().lower()
    interactive_backend = backend not in {"agg", "pdf", "ps", "svg", "cairo"}

    # If we can't show (Agg/headless), save instead.
    should_show = bool(show and interactive_backend and os.environ.get("DISPLAY"))
    if should_show:
        plt.show()
    else:
        if save_path is None:
            os.makedirs("plots", exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f"plots/tsne_{stamp}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"draw_embedding_map: saved plot to {save_path} (backend={backend}).")

    plt.close(fig)


def calculate_embedding_for_path(
    model: L.LightningModule, wav_path: str
) -> torch.Tensor:
    dataset, word, filename = extract_dataset_word_filename(wav_path)
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
    return model(mfcc.unsqueeze(0)).squeeze(0)


def get_keyword_embeddings(
    model: L.LightningModule,
    keywords: list[str],
    dataset_info,
) -> dict[str, torch.Tensor]:
    """Compute embeddings for a list of keywords using the given model.

    Returns a dict mapping keyword to embedding tensor.
    """
    keyword_embeddings: dict[str, torch.Tensor] = {}
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    with torch.no_grad():
        for keyword in keywords:
            samples = dataset_info.sample_word(keyword, n=10)
            embeddings: list[torch.Tensor] = []
            for sample in samples:
                dataset, word, filename = extract_dataset_word_filename(sample)
                mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
                embedding = model(mfcc.unsqueeze(0)).squeeze(0)
                embeddings.append(embedding)
            # Average embeddings for the keyword
            keyword_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
            keyword_embeddings[keyword] = keyword_embedding
    return keyword_embeddings


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
            # Atomic write to avoid partially-written cache files on interrupt.
            cache_dir = os.path.dirname(cached_path)
            os.makedirs(cache_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".mfcc_", suffix=".pt.tmp", dir=cache_dir
            )
            try:
                os.close(fd)
                torch.save(mfcc.cpu(), tmp_path)
                os.replace(tmp_path, cached_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
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
        cache_dir = os.path.dirname(cached_path)
        os.makedirs(cache_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=".mfcc_", suffix=".pt.tmp", dir=cache_dir
        )
        try:
            os.close(fd)
            torch.save(mfcc.cpu(), tmp_path)
            os.replace(tmp_path, cached_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
    else:
        # Cache can become corrupted if a previous run was interrupted mid-write.
        # If load fails, recompute and overwrite atomically.
        try:
            mfcc = torch.load(cached_path, map_location="cpu")
        except Exception:
            try:
                os.remove(cached_path)
            except OSError:
                pass
            mfcc = extract_mfcc(wav_path, sr, n_mfcc)

            cache_dir = os.path.dirname(cached_path)
            os.makedirs(cache_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".mfcc_", suffix=".pt.tmp", dir=cache_dir
            )
            try:
                os.close(fd)
                torch.save(mfcc.cpu(), tmp_path)
                os.replace(tmp_path, cached_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
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
        self.whole_ds = DataDataset(
            self.dataset_path,
            self.dataset_info.seen_words + self.dataset_info.unseen_words,
            n_mfcc=self.n_mfcc,
            sr=self.sr,
        )
        self.seen_ds = DataDataset(
            self.dataset_path,
            self.dataset_info.seen_words,
            n_mfcc=self.n_mfcc,
            sr=self.sr,
        )
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.seen_ds, [0.8, 0.1, 0.1]
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
