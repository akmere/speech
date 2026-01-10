import random
from typing import Any, Type
import lightning as L
import torch
import numpy as np

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
