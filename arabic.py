import csv
from functools import lru_cache
import shutil
from typing import List
from urllib.request import urlretrieve
import zipfile
from dataset import DatasetInfo, SplitName
import os

from util import extract_noise_clips

ARABIC_COMMANDS: str = "https://www.kaggle.com/api/v1/datasets/download/abdulkaderghandoura/arabic-speech-commands-dataset"
ZIP_DATA_PATH: str = "tmp/arabic.zip"
EXTRACTED_DATA_PATH: str = "tmp/arabic"

SEEN_WORDS: List[str] = [
    # zero,one,two,three,four,five,six,seven,eight,nine,right,left,up,down,forward,backward,yes,no,start,stop,enable,disable,ok,cancel,open,close,zoom in,zoom out,previous,next,send,receive,move,rotate,record,enter,digit,direction,options,undo
    "zero",
    "one",
    "eight",
    "nine",
    "right",
    "left",
    "up",
    "down",
    "yes",
    "disable",
    "move",
    "rotate",
    "ok",
    "open",
    "close",
    "zoom in",
    "zoom out",
    "previous",
    "two",
    # half
    "three",
    "four",
    "five",
    "six",
    "cancel",
    "seven",
    "send",
    "receive",
    "record",
    "enter",
    "digit",
]

UNSEEN_WORDS: List[str] = [
    "undo",
    "no",
    "start",
    "stop",
    "enable",
    # drawn embeddings
    "direction",
    "options",
    "next",
]


def prepare_arabic():
    if not os.path.exists(EXTRACTED_DATA_PATH):
        print(
            f"{EXTRACTED_DATA_PATH} does not exist, downloading from {ARABIC_COMMANDS}"
        )
        urlretrieve(ARABIC_COMMANDS, ZIP_DATA_PATH)
        print(f"Downloaded {EXTRACTED_DATA_PATH} from {ARABIC_COMMANDS}")
        with zipfile.ZipFile(ZIP_DATA_PATH, "r") as zip_ref:
            print(f"Extracting {EXTRACTED_DATA_PATH} to {EXTRACTED_DATA_PATH}")
            zip_ref.extractall(EXTRACTED_DATA_PATH)
            print(f"Files extracted to {EXTRACTED_DATA_PATH}")
        print(f"Removing {ZIP_DATA_PATH}")
        os.remove(ZIP_DATA_PATH)
        print(f"Removed {ZIP_DATA_PATH}")
        # Flatten: move {EXTRACTED_DATA_PATH}/dataset/dataset/* -> {EXTRACTED_DATA_PATH}/
        nested_root = os.path.join(EXTRACTED_DATA_PATH, "dataset", "dataset")
        if os.path.isdir(nested_root):
            for name in os.listdir(nested_root):
                src = os.path.join(nested_root, name)
                dst = os.path.join(EXTRACTED_DATA_PATH, name)

                if os.path.exists(dst):
                    raise RuntimeError(
                        f"Cannot move '{src}' -> '{dst}': destination already exists."
                    )

                shutil.move(src, dst)

            # Then remove {EXTRACTED_DATA_PATH}/dataset
            shutil.rmtree(
                os.path.join(EXTRACTED_DATA_PATH, "dataset"), ignore_errors=True
            )
        # test.cs, train.csv, val.csv are now located in EXTRACTED_DATA_PATH,
        # they all have columns file,class, modify them, removing from every line the 'dataset/' beginning the file
        for split_csv in ["test.csv", "train.csv", "val.csv"]:
            csv_path = os.path.join(EXTRACTED_DATA_PATH, split_csv)
            if os.path.isfile(csv_path):
                rows: List[dict[str, str]] = []
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rel = (row.get("file") or "").strip()
                        if rel.startswith("dataset/"):
                            rel = rel[len("dataset/") :]
                        row["file"] = rel
                        rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    fieldnames = ["file", "class"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        background_noise_dir = os.path.join(
            EXTRACTED_DATA_PATH, "background_noise", "background_noise"
        )
        output_dir = os.path.join(EXTRACTED_DATA_PATH, "background")
        extract_noise_clips(background_noise_dir, output_dir, n_clips=125)
    else:
        print(f"{EXTRACTED_DATA_PATH} exists, no download needed")
    return EXTRACTED_DATA_PATH


@lru_cache(maxsize=None)
def _load_split_sets(base_dir: str) -> tuple[set[str], set[str]]:
    """Load (test_files, val_files) as sets of relative paths from base_dir."""
    test_csv = os.path.join(base_dir, "test.csv")
    val_csv = os.path.join(base_dir, "val.csv")

    if not (os.path.isfile(test_csv) and os.path.isfile(val_csv)):
        return set(), set()

    def read_file_col(path: str) -> set[str]:
        out: set[str] = set()
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # expects columns: file,class
            for row in reader:
                rel = (row.get("file") or "").strip()
                if rel:
                    out.add(os.path.normpath(rel))
        return out

    return read_file_col(test_csv), read_file_col(val_csv)


def split_function(filename: str) -> SplitName:
    # filename is the path to the wav file
    base_dir = os.path.dirname(os.path.dirname(filename))

    test_files, val_files = _load_split_sets(base_dir)
    if not test_files and not val_files:
        return "train"

    rel = os.path.normpath(os.path.relpath(filename, base_dir))
    if rel in test_files:
        return "test"
    if rel in val_files:
        return "val"
    return "train"


arabic_dataset_info = DatasetInfo(
    prepare_arabic,
    "arabic",
    SEEN_WORDS,
    UNSEEN_WORDS,
    split_function=split_function,
)
