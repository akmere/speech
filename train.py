import argparse
import torch
import lightning as L
from model import Model
from dataset import DataModule, cache_mfccs
from speech_commands import speech_commands_dataset_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Commands a2wv (training)")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="dataset to cache mfccs for",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="dataset to cache mfccs for",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="dataset to cache mfccs for",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (set >0 after confirming stability)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="number of devices (set 1 to avoid multi-process)",
    )
    args = parser.parse_args()
    if args.cache:
        cache_mfccs(args.cache)
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Using accelerator: {accelerator}")
        N_MFCC = 40
        SR = 16000
        model = Model(N_MFCC, 0.003, 64, 0.3, True)
        dm = DataModule(
            dataset_info=speech_commands_dataset_info,
            num_workers=args.num_workers,
            n_mfcc=N_MFCC,
            sr=SR,
            k=5,
            p=3,
            steps_per_epoch=args.steps,
            val_steps_per_epoch=25,
        )
        dm.prepare_data()
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=args.devices,
            min_epochs=1,
            max_epochs=args.epochs,
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
