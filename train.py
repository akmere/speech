import argparse
import torch
import lightning as L
from model import ConvStatsPoolEncoder, Model
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
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="learning rate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gru",
        help="model type",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="margin in triplet loss",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="drouput in GRU",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k in batch triplet loss",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=3,
        help="p in batch triplet loss",
    )
    args = parser.parse_args()
    if args.cache:
        cache_mfccs(args.cache)
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Using accelerator: {accelerator}")
        N_MFCC = 40
        SR = 16000
        if args.model == "conv":
            model = ConvStatsPoolEncoder(
                N_MFCC,
                embedding_dim=64,
                lr=args.lr,
                channels=128,
                l2_normalize=True,
                margin=args.margin,
            )
        else:
            model = Model(
                N_MFCC,
                args.lr,
                embedding_dim=64,
                dropout=args.dropout,
                margin=args.margin,
                l2_normalize=True,
            )
        print("Model")
        print(model)
        dm = DataModule(
            dataset_info=speech_commands_dataset_info,
            num_workers=args.num_workers,
            n_mfcc=N_MFCC,
            sr=SR,
            k=args.k,
            p=args.p,
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
