from speech_commands import speech_commands_dataset_info
from dataset import (
    DatasetInfo,
    extract_dataset_word_filename,
    extract_or_cache_mfcc,
    draw_embedding_map_from_lists,
    get_keyword_embeddings,
)
from model import (
    ConvStatsPoolEncoder,
    KeywordEmbeddingIndex,
    calculate_missed_detection_and_false_positive_rates,
)
import torch
from speech_commands import SEEN_WORDS, UNSEEN_WORDS
from arabic import arabic_dataset_info
import random

if __name__ == "__main__":
    arabic_dataset_info.prepare_data()
    # arabic_dataset_info.test_split()
    speech_commands_dataset_info.prepare_data()
    # speech_commands_dataset_info.test_split()
    # with torch.no_grad():
    #     ckpt_path = "model.ckpt"
    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     print("ckpt keys:", ckpt.keys())
    #     print("hyper_parameters:", ckpt.get("hyper_parameters"))
    #     model = ConvStatsPoolEncoder.load_from_checkpoint(ckpt_path)
    #     model.eval()
    #     keyword_embeddings = get_keyword_embeddings(
    #         model, SEEN_WORDS, speech_commands_dataset_info
    #     )
    #     keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(keyword_embeddings)
    #     threshold: float = 0.4
    #     mdr, fpr = calculate_missed_detection_and_false_positive_rates(
    #         model,
    #         keyword_embedding_index,
    #         speech_commands_dataset_info,
    #         UNSEEN_WORDS,
    #         threshold,
    #     )
    #     print(
    #         f"At threshold {threshold:.2f}, "
    #         f"missed detection rate for unseen words: {100.0 * mdr:.2f}%, "
    #         f"false positive rate for unseen words: {100.0 * fpr:.2f}%"
    #     )
