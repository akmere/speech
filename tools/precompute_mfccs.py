import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_prep import (
    DEFAULT_AUDIO_DIR,
    LABELS_V2_NO_SILENCE,
    load_or_extract_mfcc,
    mfcc_cache_path,
)


def _list_wavs(*, audio_dir: str, words: list[str]) -> list[str]:
    wavs: list[str] = []
    for word in words:
        word_dir = os.path.join(audio_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fn in os.listdir(word_dir):
            if fn.endswith(".wav"):
                wavs.append(os.path.join(word_dir, fn))
    wavs.sort()
    return wavs


def _compute_one(
    wav_path: str,
    *,
    audio_dir: str,
    sr: int,
    n_mfcc: int,
    cache_dir: str,
    overwrite: bool,
) -> str:
    out_path = mfcc_cache_path(
        cache_dir=cache_dir,
        audio_dir=audio_dir,
        wav_path=wav_path,
        sr=sr,
        n_mfcc=n_mfcc,
    )

    if (not overwrite) and os.path.isfile(out_path):
        return "skipped"

    # Force write to cache.
    _ = load_or_extract_mfcc(
        wav_path=wav_path,
        audio_dir=audio_dir,
        sr=sr,
        n_mfcc=n_mfcc,
        cache_dir=cache_dir,
        write_cache=True,
    )
    return "wrote"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Precompute MFCC tensors (.pt) for Speech Commands"
    )
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    p.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to store cached MFCC .pt files",
    )
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mfcc", type=int, default=40)
    p.add_argument("--words", nargs="*", default=LABELS_V2_NO_SILENCE)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Process workers for MFCC extraction",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    wavs = _list_wavs(audio_dir=args.audio_dir, words=list(args.words))
    if not wavs:
        raise SystemExit("No .wav files found. Check --audio-dir/--words.")

    os.makedirs(args.cache_dir, exist_ok=True)
    prefix = os.path.join(args.cache_dir, f"sr{args.sr}_mfcc{args.n_mfcc}")
    print(f"Found {len(wavs)} wavs")
    print(f"Writing cache under: {prefix}")
    print(f"workers: {args.workers}")

    wrote = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _compute_one,
                wav_path,
                audio_dir=args.audio_dir,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                cache_dir=args.cache_dir,
                overwrite=bool(args.overwrite),
            )
            for wav_path in wavs
        ]

        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if res == "wrote":
                wrote += 1
            else:
                skipped += 1

            if i % 500 == 0 or i == len(futs):
                print(f"progress: {i}/{len(futs)} (wrote={wrote}, skipped={skipped})")

    print("Done")


if __name__ == "__main__":
    main()
