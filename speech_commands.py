import hashlib
import re
from typing import List
from urllib.request import urlretrieve
import zipfile
from dataset import DatasetInfo, SplitName
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

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, validation_percentage, testing_percentage) -> SplitName:
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    percentage_hash = (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (
        100.0 / MAX_NUM_WAVS_PER_CLASS
    )
    if percentage_hash < validation_percentage:
        result = "val"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "test"
    else:
        result = "train"
    return result


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


def split_function(filename: str) -> SplitName:
    return which_set(filename, validation_percentage=10.0, testing_percentage=10.0)


speech_commands_dataset_info = DatasetInfo(
    prepare_speech_commands, "speech_commands", SEEN_WORDS, UNSEEN_WORDS, split_function
)
