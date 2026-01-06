from typing import List
from urllib.request import urlretrieve
import zipfile
from dataset import DatasetInfo
import os

SPEECH_COMMANDS_V2_URL: str = (
    "https://www.kaggle.com/api/v1/datasets/download/yashdogra/speech-commands"
)
ZIP_DATA_PATH: str = "tmp/speech_commands.zip"
EXTRACTED_DATA_PATH: str = "tmp/speech_commands"

SEEN_WORDS: List[str] = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
]

UNSEEN_WORDS: List[str] = [
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]


def prepare_speech_commands():
    if not os.path.exists(EXTRACTED_DATA_PATH):
        print(
            f"{EXTRACTED_DATA_PATH} does not exist, downloading from {SPEECH_COMMANDS_V2_URL}"
        )
        urlretrieve(SPEECH_COMMANDS_V2_URL, ZIP_DATA_PATH)
        print(f"Downloaded {EXTRACTED_DATA_PATH} from {SPEECH_COMMANDS_V2_URL}")
        with zipfile.ZipFile(ZIP_DATA_PATH, "r") as zip_ref:
            print(f"Extracting {EXTRACTED_DATA_PATH} to {EXTRACTED_DATA_PATH}")
            zip_ref.extractall(EXTRACTED_DATA_PATH)
            print(f"Files extracted to {EXTRACTED_DATA_PATH}")
        print(f"Removing {ZIP_DATA_PATH}")
        os.remove(ZIP_DATA_PATH)
        print(f"Removed {ZIP_DATA_PATH}")
    else:
        print(f"{EXTRACTED_DATA_PATH} exists, no download needed")
    return EXTRACTED_DATA_PATH


speech_commands_dataset_info = DatasetInfo(
    prepare_speech_commands, "speech_commands", SEEN_WORDS
)
