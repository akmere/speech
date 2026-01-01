# triplet loss: L(a, p, n) = |d(a,p) - d(a,n) + margin|_+
# batch hard triplet loss: for each anchor, select hardest positive and hardest negative in the batch
# randomly sampling P classes (words) and K samples of each class, resulting in a batch size of P*K
# each sample is encoding creating P*K embeddings
# each embedding is used as anchor, hardest positive and hardest negative are selected
# audio2word-vector (a2wv) encoder: GRU network with 2 unidirectional stacked layers, a hidden state equal to the size of the embeddings of
# 64 floating point numbers and a dropout of 0.3 between the two stacked layers

import argparse
import os
import random
from typing import Final

import torch
from torch.utils.data import DataLoader

from a2wv import (
    Audio2WordVectorEncoder,
    batch_hard_triplet_loss,
    classifier,
    load_model,
    save_model,
)
from data_prep import (
    DEFAULT_AUDIO_DIR,
    LABELS_V2_NO_SILENCE,
    PKBatchSampler,
    SpeechCommandsDataset,
    collate_pad,
    extract_mfcc,
    read_split_list,
)


AUDIO_DIR: Final[str] = DEFAULT_AUDIO_DIR
DEFAULT_WORDS: Final[list[str]] = LABELS_V2_NO_SILENCE


@torch.no_grad()
def _build_prototypes(
    *,
    model: Audio2WordVectorEncoder,
    audio_dir: str,
    words: list[str],
    sr: int,
    n_mfcc: int,
    per_class: int,
    device: torch.device,
) -> torch.Tensor:
    # Prototypes are mean embeddings per class: (C, D)
    prototypes: list[torch.Tensor] = []
    for word in words:
        word_dir = os.path.join(audio_dir, word)
        if not os.path.isdir(word_dir):
            raise FileNotFoundError(f"Missing word folder: {word_dir}")

        wavs = sorted(
            os.path.join(word_dir, fn)
            for fn in os.listdir(word_dir)
            if fn.endswith(".wav")
        )
        if not wavs:
            raise ValueError(f"No .wav files found for word '{word}' under {word_dir}")

        wavs = wavs[: max(1, per_class)]
        embs: list[torch.Tensor] = []
        for wav_path in wavs:
            x = extract_mfcc(wav_path, sr=sr, n_mfcc=n_mfcc).to(device)
            xb = x.unsqueeze(0)  # (1, T, F)
            lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=device)
            z = model(xb, lengths=lengths)[0]  # (D,)
            embs.append(z)

        proto = torch.stack(embs, dim=0).mean(dim=0)
        proto = torch.nn.functional.normalize(proto, p=2, dim=0)
        prototypes.append(proto)

    return torch.stack(prototypes, dim=0)


@torch.no_grad()
def test_audio_file(
    *,
    audio_path: str,
    audio_dir: str,
    model_path: str,
    words: list[str],
    sr: int = 16000,
    threshold: float = 1.0,
    prototypes_per_class: int = 5,
    device: str | None = None,
) -> int:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model(model_path, dev)
    n_mfcc = int(model.gru.input_size)

    prototypes = _build_prototypes(
        model=model,
        audio_dir=audio_dir,
        words=words,
        sr=sr,
        n_mfcc=n_mfcc,
        per_class=prototypes_per_class,
        device=dev,
    )

    x = extract_mfcc(audio_path, sr=sr, n_mfcc=n_mfcc).to(dev)
    xb = x.unsqueeze(0)  # (1, T, F)

    def f(inp: torch.Tensor) -> torch.Tensor:
        if inp.ndim != 3 or inp.shape[0] != 1:
            raise ValueError("Expected input shape (1, T, F) for classification")
        lengths = torch.tensor([inp.shape[1]], dtype=torch.long, device=inp.device)
        return model(inp, lengths=lengths)

    pred = classifier(xb, f=f, prototypes=prototypes, t=threshold)

    # For user feedback, also compute the closest distance.
    z = f(xb)[0]
    distances = torch.linalg.vector_norm(prototypes - z.unsqueeze(0), ord=2, dim=1)
    min_dist, argmin = torch.min(distances, dim=0)

    if pred == -1:
        print(
            f"prediction: <rejected> (closest='{words[int(argmin.item())]}', dist={float(min_dist.item()):.4f}, threshold={threshold})"
        )
    else:
        print(
            f"prediction: {words[pred]} (dist={float(min_dist.item()):.4f}, threshold={threshold})"
        )
    # Also print distances to all words (sorted by closest first).
    dist_items = [(words[i], float(distances[i].item())) for i in range(len(words))]
    dist_items.sort(key=lambda x: x[1])
    print("distances (closest -> farthest):")
    for w, d in dist_items:
        print(f"  {w}: {d:.4f}")
    return pred


def train_model(
    *,
    audio_dir: str = AUDIO_DIR,
    words: list[str] | None = DEFAULT_WORDS,
    n_mfcc: int = 40,
    sr: int = 16000,
    embedding_dim: int = 64,
    P: int = 5,
    K: int = 3,
    steps_per_epoch: int = 100,
    epochs: int = 5,
    lr: float = 1e-3,
    margin: float = 1.0,
    device: str | None = None,
    model_path: str = "a2wv.pt",
    seed: int = 0,
    unique_speakers: bool = False,
    max_items_per_word: int | None = None,
    val_steps_per_epoch: int = 25,
    use_official_validation_split: bool = True,
) -> str:
    torch.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    val_relpaths: set[str] = set()
    test_relpaths: set[str] = set()
    if use_official_validation_split:
        val_relpaths = read_split_list(os.path.join(audio_dir, "validation_list.txt"))
        test_relpaths = read_split_list(os.path.join(audio_dir, "testing_list.txt"))
        if not val_relpaths:
            print(
                "validation_list.txt not found or empty; falling back to train-only loss"
            )

    train_exclude = (val_relpaths | test_relpaths) if val_relpaths else None

    train_ds = SpeechCommandsDataset(
        audio_dir=audio_dir,
        words=words,
        sr=sr,
        n_mfcc=n_mfcc,
        unique_speakers=unique_speakers,
        max_items_per_word=max_items_per_word,
        exclude_relpaths=train_exclude,
    )

    val_ds: SpeechCommandsDataset | None = None
    if val_relpaths:
        val_ds = SpeechCommandsDataset(
            audio_dir=audio_dir,
            words=words,
            sr=sr,
            n_mfcc=n_mfcc,
            unique_speakers=False,
            max_items_per_word=None,
            include_relpaths=val_relpaths,
        )

    train_sampler = PKBatchSampler(
        train_ds.label_to_indices,
        P=P,
        K=K,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_pad,
        pin_memory=(dev.type == "cuda"),
    )

    val_loader: DataLoader | None = None
    val_sampler: PKBatchSampler | None = None
    if val_ds is not None:
        val_sampler = PKBatchSampler(
            val_ds.label_to_indices,
            P=P,
            K=K,
            steps_per_epoch=val_steps_per_epoch,
            seed=seed + 999,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=0,
            collate_fn=collate_pad,
            pin_memory=(dev.type == "cuda"),
        )

    print("Len train_ds:", len(train_ds))
    # 84843
    print("Len val_ds:", len(val_ds) if val_ds is not None else 0)
    # 9981
    print("Len test_ds:", len(test_relpaths) if test_relpaths else 0)
    # 11005
    model = Audio2WordVectorEncoder(input_dim=n_mfcc, embedding_dim=embedding_dim)
    model.to(dev)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        # Change sampling each epoch for better coverage
        train_sampler.seed = seed + epoch

        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x = batch.x.to(dev, non_blocking=True)
            lengths = batch.lengths.to(dev)
            labels = batch.labels.to(dev)

            opt.zero_grad(set_to_none=True)
            emb = model(x, lengths=lengths)
            loss = batch_hard_triplet_loss(emb, labels, margin=margin)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if step % 10 == 0:
                avg = running / step
                print(
                    f"epoch {epoch}/{epochs} step {step}/{steps_per_epoch} loss={avg:.4f}"
                )

        train_loss = running / max(1, steps_per_epoch)

        val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for vstep, vbatch in enumerate(val_loader, start=1):
                    vx = vbatch.x.to(dev, non_blocking=True)
                    vlengths = vbatch.lengths.to(dev)
                    vlabels = vbatch.labels.to(dev)
                    vemb = model(vx, lengths=vlengths)
                    vloss = batch_hard_triplet_loss(vemb, vlabels, margin=margin)
                    val_running += float(vloss.item())
            val_loss = val_running / max(1, val_steps_per_epoch)
            print(
                f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                save_model(model, model_path)
                print(f"saved best model to {model_path} (val_loss={best_val:.4f})")
        else:
            print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f}")
            # fall back to saving best train loss
            if train_loss < best_val:
                best_val = train_loss
                save_model(model, model_path)
                print(f"saved best model to {model_path} (train_loss={best_val:.4f})")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Speech Commands a2wv (training)")
    parser.add_argument("--train", action="store_true", help="Train the encoder")
    parser.add_argument(
        "--test-audio",
        type=str,
        default=None,
        help="Classify a single .wav file using the saved model and dataset prototypes",
    )
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--model-path", default="a2wv.pt")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device for training/inference (e.g. 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate for loading audio during inference/prototype building",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Rejection threshold t for classifier (smaller = stricter)",
    )
    parser.add_argument(
        "--prototypes-per-class",
        type=int,
        default=5,
        help="How many wavs per class to average into prototypes",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=5657)
    parser.add_argument("--P", type=int, default=5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unique-speakers", action="store_true")
    parser.add_argument("--max-items-per-word", type=int, default=None)
    parser.add_argument("--val-steps-per-epoch", type=int, default=666)
    args = parser.parse_args()

    if args.train:
        train_model(
            audio_dir=args.audio_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            val_steps_per_epoch=args.val_steps_per_epoch,
            P=args.P,
            K=args.K,
            lr=args.lr,
            margin=args.margin,
            seed=args.seed,
            unique_speakers=args.unique_speakers,
            max_items_per_word=args.max_items_per_word,
        )
        return

    if args.test_audio is not None:
        test_audio_file(
            audio_path=args.test_audio,
            audio_dir=args.audio_dir,
            model_path=args.model_path,
            words=DEFAULT_WORDS,
            sr=args.sr,
            threshold=args.threshold,
            prototypes_per_class=args.prototypes_per_class,
            device=args.device,
        )
        return

    print("Use --train to train a model, or --test-audio to classify a file.")


if __name__ == "__main__":
    main()
