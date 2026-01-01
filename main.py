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
import shutil
import subprocess
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
    load_or_extract_mfcc,
    mfcc_cache_path,
    read_split_list,
)


AUDIO_DIR: Final[str] = DEFAULT_AUDIO_DIR
DEFAULT_WORDS: Final[list[str]] = LABELS_V2_NO_SILENCE


def embedding_cache_path(
    *,
    cache_dir: str,
    audio_dir: str,
    wav_path: str,
) -> str:
    rel = os.path.relpath(wav_path, audio_dir).replace("\\", "/")
    rel_no_ext, _ext = os.path.splitext(rel)
    return os.path.join(cache_dir, rel_no_ext + ".pt")


def precompute_all_mfccs(
    *,
    audio_dir: str,
    words: list[str],
    sr: int,
    n_mfcc: int,
    mfcc_cache_dir: str,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Precompute MFCC tensors to disk.

    Writes `torch.Tensor` MFCCs to:
      {mfcc_cache_dir}/sr{sr}_mfcc{n_mfcc}/<relpath-without-ext>.pt

    Returns (wrote, skipped).
    """
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    os.makedirs(mfcc_cache_dir, exist_ok=True)

    wavs: list[str] = []
    for word in words:
        word_dir = os.path.join(audio_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fn in os.listdir(word_dir):
            if fn.endswith(".wav"):
                wavs.append(os.path.join(word_dir, fn))
    wavs.sort()

    wrote = 0
    skipped = 0
    total = len(wavs)
    if total == 0:
        return (0, 0)

    for i, wav_path in enumerate(wavs, start=1):
        out_path = mfcc_cache_path(
            cache_dir=mfcc_cache_dir,
            audio_dir=audio_dir,
            wav_path=wav_path,
            sr=sr,
            n_mfcc=n_mfcc,
        )
        if (not overwrite) and os.path.isfile(out_path):
            skipped += 1
            continue

        _ = load_or_extract_mfcc(
            wav_path=wav_path,
            audio_dir=audio_dir,
            sr=sr,
            n_mfcc=n_mfcc,
            cache_dir=mfcc_cache_dir,
            write_cache=True,
        )
        wrote += 1

        if i % 500 == 0 or i == total:
            print(f"mfcc progress: {i}/{total} (wrote={wrote}, skipped={skipped})")

    return (wrote, skipped)


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
    mfcc_cache_dir: str | None,
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
            x = load_or_extract_mfcc(
                wav_path=wav_path,
                audio_dir=audio_dir,
                sr=sr,
                n_mfcc=n_mfcc,
                cache_dir=mfcc_cache_dir,
                write_cache=False,
            ).to(device)
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
    mfcc_cache_dir: str | None = None,
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
        mfcc_cache_dir=mfcc_cache_dir,
    )

    x = load_or_extract_mfcc(
        wav_path=audio_path,
        audio_dir=audio_dir,
        sr=sr,
        n_mfcc=n_mfcc,
        cache_dir=mfcc_cache_dir,
        write_cache=False,
    ).to(dev)
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
    mfcc_cache_dir: str | None = None,
    mfcc_cache_write: bool = False,
) -> str:
    torch.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print("device: ", device)

    val_relpaths: set[str] = set()
    test_relpaths: set[str] = set()
    if use_official_validation_split:
        val_relpaths = read_split_list(os.path.join(audio_dir, "validation_list.txt"))
        test_relpaths = read_split_list(os.path.join(audio_dir, "testing_list.txt"))
        if not val_relpaths and not test_relpaths:
            print(
                "validation_list.txt/testing_list.txt not found or empty; falling back to train-only loss"
            )

    # Exclude whatever official splits we actually have (val and/or test).
    train_exclude = (
        (val_relpaths | test_relpaths) if (val_relpaths or test_relpaths) else None
    )

    train_ds = SpeechCommandsDataset(
        audio_dir=audio_dir,
        words=words,
        sr=sr,
        n_mfcc=n_mfcc,
        mfcc_cache_dir=mfcc_cache_dir,
        mfcc_cache_write=mfcc_cache_write,
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
            mfcc_cache_dir=mfcc_cache_dir,
            mfcc_cache_write=False,
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
        num_workers=min(8, os.cpu_count() or 0),
        collate_fn=collate_pad,
        pin_memory=(dev.type == "cuda"),
    )

    val_loader: DataLoader | None = None
    val_sampler: PKBatchSampler | None = None
    if val_ds is not None:
        # Guard: official val split might not contain enough classes for requested P.
        if len(val_ds.label_to_indices) < P:
            print(
                f"Warning: validation split has only {len(val_ds.label_to_indices)} classes, "
                f"but P={P}. Disabling validation loader."
            )
            val_ds = None
        else:
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
                num_workers=min(8, os.cpu_count() or 0),
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
            # collapse checks (variation across samples, not across dimensions)
            batch_std = float(
                emb.detach().float().std(dim=0).mean().item()
            )  # ~0 if all samples same
            mean_pair_dist = float(
                torch.cdist(emb.detach(), emb.detach()).mean().item()
            )
            loss = batch_hard_triplet_loss(emb, labels, margin=margin)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if step % 10 == 0:
                avg = running / step
                emb_std = float(emb.detach().float().std().item())
                emb_norm_mean = float(emb.detach().norm(p=2, dim=1).mean().item())
                print(
                    f"epoch {epoch}/{epochs} step {step}/{steps_per_epoch} "
                    f"loss_cur={loss.item():.6f} loss_avg={avg:.6f} "
                    f"emb_std={emb_std:.6f} emb_norm_mean={emb_norm_mean:.6f} "
                    f"batch_std={batch_std:.6e} mean_pair_dist={mean_pair_dist:.6e}"
                )

        train_loss = running / max(1, steps_per_epoch)

        val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            vsteps = 0
            with torch.no_grad():
                for vsteps, vbatch in enumerate(val_loader, start=1):
                    vx = vbatch.x.to(dev, non_blocking=True)
                    vlengths = vbatch.lengths.to(dev)
                    vlabels = vbatch.labels.to(dev)
                    vemb = model(vx, lengths=vlengths)
                    vloss = batch_hard_triplet_loss(vemb, vlabels, margin=margin)
                    val_running += float(vloss.item())
            val_loss = val_running / max(1, vsteps)
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


def _pick_audio_player() -> list[str] | None:
    # Prefer ffplay (from ffmpeg) since it's widely available and doesn't require a GUI.
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    # ALSA
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    # PulseAudio
    if shutil.which("paplay"):
        return ["paplay"]
    return None


def _audit_training_audio(
    *,
    audio_dir: str,
    words: list[str],
    sr: int,
    n_mfcc: int,
    P: int,
    K: int,
    steps_per_epoch: int,
    seed: int,
    unique_speakers: bool,
    max_items_per_word: int | None,
    use_official_validation_split: bool,
    batches: int,
    play: bool,
    interactive: bool,
    mfcc_cache_dir: str | None,
) -> None:
    val_relpaths: set[str] = set()
    test_relpaths: set[str] = set()
    if use_official_validation_split:
        val_relpaths = read_split_list(os.path.join(audio_dir, "validation_list.txt"))
        test_relpaths = read_split_list(os.path.join(audio_dir, "testing_list.txt"))
        if not val_relpaths and not test_relpaths:
            print(
                "validation_list.txt/testing_list.txt not found or empty; audit will use full folder dataset"
            )

    # Exclude whatever official splits we actually have (val and/or test).
    train_exclude = (
        (val_relpaths | test_relpaths) if (val_relpaths or test_relpaths) else None
    )

    train_ds = SpeechCommandsDataset(
        audio_dir=audio_dir,
        words=words,
        sr=sr,
        n_mfcc=n_mfcc,
        mfcc_cache_dir=mfcc_cache_dir,
        mfcc_cache_write=False,
        unique_speakers=unique_speakers,
        max_items_per_word=max_items_per_word,
        exclude_relpaths=train_exclude,
    )

    sampler = PKBatchSampler(
        train_ds.label_to_indices,
        P=P,
        K=K,
        steps_per_epoch=max(1, steps_per_epoch),
        seed=seed,
    )

    player = _pick_audio_player() if play else None
    if play and player is None:
        print("No audio player found (tried: ffplay, aplay, paplay).")
        print("Install ffmpeg or alsa-utils, or run without --audit-play.")

    print("Audit dataset:")
    print(f"  audio_dir: {audio_dir}")
    print(f"  items: {len(train_ds)}")
    print(f"  words: {len(train_ds.words)}")
    print(f"  sampling: P={P}, K={K} (batch size={P * K})")
    print(f"  exclude official val/test: {bool(train_exclude)}")

    shown = 0
    mismatches = 0
    for b, batch_indices in enumerate(iter(sampler), start=1):
        if b > batches:
            break

        print(f"\n=== audit batch {b}/{batches} ===")
        for j, idx in enumerate(batch_indices, start=1):
            wav_path = train_ds.wav_path(idx)
            _x, label = train_ds[idx]
            word = train_ds.word_for_label(label)

            rel = os.path.relpath(wav_path, audio_dir).replace("\\", "/")
            folder = rel.split("/", 1)[0] if "/" in rel else ""
            ok = folder == word
            if not ok:
                mismatches += 1

            status = "OK" if ok else "MISMATCH"
            print(
                f"{j:02d}. label={int(label):02d} word='{word}' folder='{folder}' [{status}]"
            )
            print(f"    {rel}")

            if player is not None:
                if interactive:
                    ans = input("    Press Enter to play (q to quit): ").strip().lower()
                    if ans in {"q", "quit"}:
                        print("Stopping audit.")
                        print(f"Summary: shown={shown} mismatches={mismatches}")
                        return
                subprocess.run(player + [wav_path], check=False)

            shown += 1

    print(f"\nSummary: shown={shown} mismatches={mismatches}")


def main():
    parser = argparse.ArgumentParser(description="Speech Commands a2wv (training)")
    parser.add_argument("--train", action="store_true", help="Train the encoder")
    parser.add_argument(
        "--precompute-mfccs",
        action="store_true",
        help="Precompute MFCC cache files under --mfcc-cache-dir",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Preview which wavs/labels are used for training (optionally play audio)",
    )
    parser.add_argument(
        "--audit-batches",
        type=int,
        default=1,
        help="How many sampled PK batches to preview during --audit",
    )
    parser.add_argument(
        "--audit-play",
        action="store_true",
        help="Play each wav during --audit (requires ffplay/aplay/paplay)",
    )
    parser.add_argument(
        "--audit-non-interactive",
        action="store_true",
        help="Don't prompt between clips when using --audit-play",
    )
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
    parser.add_argument(
        "--mfcc-cache-dir",
        type=str,
        default="mfcc_cache",
        help="Optional directory containing precomputed MFCC .pt files (see tools/precompute_mfccs.py)",
    )
    parser.add_argument(
        "--mfcc-n-mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients to precompute when using --precompute-mfccs",
    )
    parser.add_argument(
        "--mfcc-cache-overwrite",
        action="store_true",
        help="Overwrite existing cached MFCC files when using --precompute-mfccs",
    )
    parser.add_argument(
        "--mfcc-cache-write",
        default=True,
        action="store_true",
        help="If set, writes MFCCs into --mfcc-cache-dir on cache misses during training",
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
    parser.add_argument(
        "--use-official-validation-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude official val/test lists from training/audit",
    )
    args = parser.parse_args()

    if args.precompute_mfccs:
        wrote, skipped = precompute_all_mfccs(
            audio_dir=args.audio_dir,
            words=DEFAULT_WORDS,
            sr=args.sr,
            n_mfcc=int(args.mfcc_n_mfcc),
            mfcc_cache_dir=args.mfcc_cache_dir,
            overwrite=bool(args.mfcc_cache_overwrite),
        )
        print(f"mfcc cache done: wrote={wrote} skipped={skipped}")
        return

    if args.audit:
        _audit_training_audio(
            audio_dir=args.audio_dir,
            words=DEFAULT_WORDS,
            sr=args.sr,
            n_mfcc=40,
            P=args.P,
            K=args.K,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
            unique_speakers=args.unique_speakers,
            max_items_per_word=args.max_items_per_word,
            use_official_validation_split=args.use_official_validation_split,
            batches=max(1, int(args.audit_batches)),
            play=bool(args.audit_play),
            interactive=not bool(args.audit_non_interactive),
            mfcc_cache_dir=args.mfcc_cache_dir,
        )
        return

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
            use_official_validation_split=args.use_official_validation_split,
            mfcc_cache_dir=args.mfcc_cache_dir,
            mfcc_cache_write=bool(args.mfcc_cache_write),
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
            mfcc_cache_dir=args.mfcc_cache_dir,
        )
        return

    print("Use --train to train a model, or --test-audio to classify a file.")


if __name__ == "__main__":
    main()
