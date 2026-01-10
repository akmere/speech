import random
from typing import Any, Type
import lightning as L
import torch
import numpy as np
import wave
import contextlib
import os


from dataset import DatasetInfo, extract_dataset_word_filename, extract_or_cache_mfcc


def _as_dict(x: Any) -> dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    # e.g. argparse.Namespace or SimpleNamespace
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {}


def load_lightning_module(
    cls: Type[L.LightningModule], path: str, **kwargs
) -> L.LightningModule:
    """
    Load a specific LightningModule subclass from a checkpoint.

    Behavior:
    1) Try Lightning's native `load_from_checkpoint`.
    2) If that fails due to missing init args, manually:
       - read `hyper_parameters` from the checkpoint
       - instantiate `cls(**hyper_parameters, **kwargs)`
       - load `state_dict`
    """
    try:
        return cls.load_from_checkpoint(path, **kwargs)
    except TypeError as e:
        # Fallback for checkpoints that require init args and can't be inferred automatically
        ckpt = torch.load(path, map_location="cpu")

        hparams = _as_dict(ckpt.get("hyper_parameters"))
        if not hparams:
            raise TypeError(
                f"Checkpoint at {path} has no 'hyper_parameters', so {cls.__name__} "
                f"cannot be constructed without explicit init args. Original error: {e}"
            ) from e

        # Allow caller kwargs to override checkpoint hyperparams
        init_kwargs = {**hparams, **kwargs}

        model = cls(**init_kwargs)

        state_dict = ckpt.get("state_dict")
        if state_dict is None:
            raise KeyError(f"Checkpoint at {path} has no 'state_dict' key.")

        # allow passing strict=False if needed
        strict = init_kwargs.pop("strict", True) if "strict" in init_kwargs else True
        model.load_state_dict(state_dict, strict=strict)
        return model


def pad_wav(wav, wav_max_length, pad=0):
    """Pads audio wave sequence to be `wav_max_length` long."""
    dim = wav.shape[1]
    padded = np.zeros((wav_max_length, dim)) + pad
    if len(wav) > wav_max_length:
        wav = wav[:wav_max_length]
    length = len(wav)
    padded[:length, :] = wav
    return padded, length


def extract_noise_clips(background_noise_dir: str, output_dir: str, n_clips: int):
    os.makedirs(output_dir, exist_ok=True)
    if n_clips <= 0:
        return

    # --- 1) Scan files to learn capacity (how many full 1-second clips each can provide) ---
    wav_files: list[str] = [
        f for f in os.listdir(background_noise_dir) if f.endswith(".wav")
    ]
    if not wav_files:
        print(f"Warning: no .wav files found in {background_noise_dir}.")
        return

    file_metas: list[dict[str, Any]] = []
    for filename in wav_files:
        file_path = os.path.join(background_noise_dir, filename)
        try:
            with contextlib.closing(wave.open(file_path, "r")) as wf:
                framerate = wf.getframerate()
                num_frames = wf.getnframes()
                if framerate <= 0 or num_frames <= 0:
                    continue
                num_seconds = num_frames // framerate
                if num_seconds <= 0:
                    continue
                file_metas.append(
                    {
                        "filename": filename,
                        "path": file_path,
                        "num_seconds": int(num_seconds),
                    }
                )
        except wave.Error:
            continue

    if not file_metas:
        print(f"Warning: no usable .wav files found in {background_noise_dir}.")
        return

    capacities = [m["num_seconds"] for m in file_metas]
    total_capacity = sum(capacities)
    if total_capacity <= 0:
        print(f"Warning: total capacity is 0 in {background_noise_dir}.")
        return

    if n_clips > total_capacity:
        print(
            f"Warning: requested {n_clips} clips, but only {total_capacity} full "
            f"1-second clips are available. Extracting {total_capacity}."
        )
    clips_to_allocate = min(n_clips, total_capacity)

    # --- 2) Randomly allocate how many clips to take from each file (decided at the start) ---
    remaining_caps = capacities[:]  # decreases as we allocate
    alloc = [0] * len(file_metas)

    # Each "allocation draw" picks a file proportional to its remaining capacity,
    # ensuring alloc[i] never exceeds that file's capacity.
    for _ in range(clips_to_allocate):
        idx = random.choices(range(len(file_metas)), weights=remaining_caps, k=1)[0]
        alloc[idx] += 1
        remaining_caps[idx] -= 1

    # Optional: randomize file processing order so output indices ("i") come from random files.
    indices = list(range(len(file_metas)))
    random.shuffle(indices)

    # --- 3) Extract allocated clips per file, with random seconds per file ---
    extracted = 0
    for idx in indices:
        target = alloc[idx]
        if target <= 0:
            continue

        filename = file_metas[idx]["filename"]
        file_path = file_metas[idx]["path"]
        num_seconds = file_metas[idx]["num_seconds"]

        with contextlib.closing(wave.open(file_path, "r")) as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            num_frames = wf.getnframes()

            if framerate <= 0 or num_frames <= 0:
                continue

            # Determine dtype from sample width.
            if sample_width == 1:
                dtype = np.uint8  # 8-bit PCM is typically unsigned
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")

            audio_data = wf.readframes(num_frames)
            audio_array = np.frombuffer(audio_data, dtype=dtype)

            # Convert to frames x channels (wave data is interleaved for multi-channel).
            if num_channels > 1:
                audio_array = audio_array.reshape(-1, num_channels)
            else:
                audio_array = audio_array.reshape(-1, 1)

            # Recompute to be safe (should match scan, but avoids edge cases).
            num_seconds_actual = audio_array.shape[0] // framerate
            if num_seconds_actual <= 0:
                continue
            num_seconds = min(num_seconds, num_seconds_actual)

            k = min(target, num_seconds)
            chosen_seconds = random.sample(range(num_seconds), k=k)  # random seconds

            base = os.path.splitext(filename)[0]
            for sec in chosen_seconds:
                start = sec * framerate
                end = start + framerate
                clip_frames = audio_array[start:end, :]  # (framerate, channels)

                clip_filename = f"{base}_clip_{extracted}.wav"
                clip_path = os.path.join(output_dir, clip_filename)

                with wave.open(clip_path, "w") as clip_wf:
                    clip_wf.setnchannels(num_channels)
                    clip_wf.setsampwidth(sample_width)
                    clip_wf.setframerate(framerate)
                    clip_wf.writeframes(clip_frames.reshape(-1).tobytes())

                extracted += 1
                print(f"Extracted clip: {clip_path}")

    if extracted < n_clips:
        print(f"Warning: requested {n_clips} clips, but only extracted {extracted}.")
