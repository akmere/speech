from speech_commands import speech_commands_dataset_info
from dataset import (
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
import random


if __name__ == "__main__":
    with torch.no_grad():
        ckpt_path = "model.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("ckpt keys:", ckpt.keys())
        print("hyper_parameters:", ckpt.get("hyper_parameters"))
        model = ConvStatsPoolEncoder.load_from_checkpoint(ckpt_path)
        model.eval()
        keyword_embeddings = get_keyword_embeddings(
            model, SEEN_WORDS, speech_commands_dataset_info
        )
        keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(keyword_embeddings)
        samples: list[str] = []
        k_seen_words: int = 5
        if len(SEEN_WORDS) < k_seen_words:
            k_seen_words = len(SEEN_WORDS)
        k_unseen_words: int = 5
        if len(UNSEEN_WORDS) < k_unseen_words:
            k_unseen_words = len(UNSEEN_WORDS)
        for word in random.sample(SEEN_WORDS, k=k_seen_words) + random.sample(
            UNSEEN_WORDS, k=k_unseen_words
        ):
            samples.extend(speech_commands_dataset_info.sample_word(word, n=10))
        embeddings: list[torch.Tensor] = []
        labels: list[str] = []
        for sample in samples:
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename)
            embedding = model(mfcc.unsqueeze(0)).squeeze(0)
            print(filename)
            print(mfcc)
            print(embedding)
            embeddings.append(embedding)
            if word in SEEN_WORDS:
                labels.append(f"{word}")
            else:
                labels.append(f"{word} (unseen)")
        draw_embedding_map_from_lists(
            embeddings,
            labels,
            show=False,
            save_path="plots/embedding_map.png",
        )
        threshold: float = 0.4
        mdr, fpr = calculate_missed_detection_and_false_positive_rates(
            model,
            keyword_embedding_index,
            speech_commands_dataset_info,
            UNSEEN_WORDS,
            threshold,
        )
        print(
            f"At threshold {threshold:.2f}, "
            f"missed detection rate for unseen words: {100.0 * mdr:.2f}%, "
            f"false positive rate for unseen words: {100.0 * fpr:.2f}%"
        )
