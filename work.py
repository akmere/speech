from speech_commands import speech_commands_dataset_info
from dataset import extract_dataset_word_filename, extract_or_cache_mfcc
from util import load_lightning_module
from model import GRU
import torch

if __name__ == "__main__":
    with torch.no_grad():
        ckpt_path = "trained_models/gru/epoch=3-step=40.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("ckpt keys:", ckpt.keys())
        print("hyper_parameters:", ckpt.get("hyper_parameters"))
        model = load_lightning_module(GRU, "trained_models/gru/epoch=3-step=40.ckpt")
        model.eval()
        samples = speech_commands_dataset_info.sample_word("zero", 10)
        for sample in samples:
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename)
            embedding = model(mfcc.unsqueeze(0)).squeeze(0)
            print(filename)
            print(mfcc)
            print(embedding)
