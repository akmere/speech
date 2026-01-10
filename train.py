import argparse
import torch
import lightning as L
from model import ConvStatsPoolEncoder, GRUEncoder
from dataset import DataModule, cache_mfccs
from speech_commands import speech_commands_dataset_info
from arabic import arabic_dataset_info
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os

MODEL_PATH: str = "trained_models/"


if __name__ == "__main__":
    os.makedirs(MODEL_PATH, exist_ok=True)
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
        "--num-workers",
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
        default=3,
        help="k in batch triplet loss",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=5,
        help="p in batch triplet loss",
    )
    parser.add_argument(
        "--l2",
        action="store_true",
        default=None,
        help="l2 normalization",
    )
    parser.add_argument(
        "--det-curves",
        action="store_true",
        default=None,
        help="compute DET curves after validation epoch",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="threshold for classification",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        default=None,
        help="disable logging",
    )
    parser.add_argument(
        "--fix-length",
        type=int,
        default=32,
        help="fix length of input sequences",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="dataset to use: 'speech_commands' or 'arabic'",
    )
    args = parser.parse_args()
    if args.cache:
        cache_mfccs(args.cache)
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        print(f"Using accelerator: {accelerator}")
        N_MFCC = 40
        SR = 16000
        config = {
            "lr": args.lr,
            "embedding_dim": 64,
            "margin": args.margin,
            "n_mfcc": N_MFCC,
            "steps_per_epoch": args.steps,
            "val_steps_per_epoch": 25,
            "k": args.k,
            "p": args.p,
            "l2": args.l2 is not None,
            "det_curves": args.det_curves is not None,
            "threshold": args.threshold,
            "fix_length": args.fix_length,
            "model": args.model,
            "dataset": args.dataset,
        }
        if config["dataset"] == "arabic":
            dataset_info = arabic_dataset_info
            embeddings_words = [
                "down",
                "yes",
                "disable",
                "move",
                "rotate",
                "undo",
                "no",
                "start",
                "stop",
                "enable",
            ]
        elif config["dataset"] == "speech_commands":
            dataset_info = speech_commands_dataset_info
            embeddings_words = [
                "sheila",
                "seven",
                "eight",
                "no",
                "down",
                "follow",
                "learn",
                "visual",
                "right",
                "on",
            ]

        else:
            raise ValueError(
                f"Unknown dataset: {config['dataset']}. "
                "Please specify --dataset as 'speech_commands' or 'arabic'."
            )
        if config["model"] == "conv":
            config.update({"channels": 128})
            model = ConvStatsPoolEncoder(
                input_dim=config["n_mfcc"],
                embedding_dim=config["embedding_dim"],
                lr=config["lr"],
                channels=config["channels"],
                l2_normalize=config["l2"],
                margin=config["margin"],
                threshold=config["threshold"],
                dataset_info=dataset_info,
                embeddings_words=embeddings_words,
                det_curves=config["det_curves"],
            )
        else:
            config.update({"dropout": args.dropout})
            model = GRUEncoder(
                input_dim=config["n_mfcc"],
                lr=config["lr"],
                embedding_dim=config["embedding_dim"],
                dropout=config["dropout"],
                margin=config["margin"],
                l2_normalize=config["l2"],
                threshold=config["threshold"],
                dataset_info=dataset_info,
                embeddings_words=embeddings_words,
                det_curves=config["det_curves"],
            )
        print("Model")
        print(model)
        print("config")
        print(config)
        dm = DataModule(
            dataset_info=dataset_info,
            num_workers=args.num_workers,
            n_mfcc=config["n_mfcc"],
            sr=SR,
            k=config["k"],
            p=config["p"],
            steps_per_epoch=config["steps_per_epoch"],
            val_steps_per_epoch=config["val_steps_per_epoch"],
            fixed_length=config["fix_length"],
        )
        dm.prepare_data()
        ckpt_dir: str = f"{config['model']}-{config['dataset']}"
        # Let Lightning manage the W&B run; it ensures metrics/images logged via
        # `self.logger` go to W&B.
        wandb_logger = WandbLogger(
            project="ssp2025p",
            name=ckpt_dir,
            config=config,
            sync_tensorboard=False,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(f"{MODEL_PATH}", ckpt_dir),
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=args.devices,
            min_epochs=1,
            max_epochs=args.epochs,
            logger=wandb_logger if not args.nolog else False,
            enable_checkpointing=not args.nolog,
            callbacks=[checkpoint_callback] if not args.nolog else [],
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule=dm)
        # trainer.validate(model, dm)
        trainer.test(model, datamodule=dm)
        # get some random samples from test_ds
        # dm.setup(stage="train")
        # for x, lengths, labels in dm.train_dataloader():
        #     print("Sample batch from test set")
        #     print("x:", x.shape)
        #     print("labels:", labels)
        #     print("Predictions:")

        #     model.eval()
        #     with torch.inference_mode():
        #         # Run a direct forward pass so we get a Tensor (trainer.predict returns a list/None).
        #         embeddings = model(x, lengths)

        #     print("embeddings:", embeddings)
        #     draw_embedding_map(
        #         embeddings,
        #         labels,
        #         show=False,
        #         save_path="plots/latest_tsne.png",
        #         labels_to_strings=list(
        #             seen_word + " (seen)" for seen_word in dm.dataset_info.words
        #         )
        #         + list(
        #             unseen_word + " (unseen)"
        #             for unseen_word in dm.dataset_info.unseen_words
        #         ),
        #     )
        #     break
        # print(
        #     trainer.predict(
        #         model,
        #         torch.tensor(
        #             extract_or_cache_mfcc(
        #                 "speech_commands", "bed", "0a196374_nohash_0.wav"
        #             )
        #         ).unsqueeze(0),
        #     )
        # )
