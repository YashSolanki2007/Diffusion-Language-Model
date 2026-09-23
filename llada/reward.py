from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


def _ngrams(text: str, order: int) -> Counter[str]:
    if len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def character_ngram_f1(candidate: str, reference: str, max_order: int = 4) -> float:
    """A compact character-level analogue of chrF, bounded to [0, 1]."""
    scores = []
    for order in range(1, max_order + 1):
        candidate_counts = _ngrams(candidate, order)
        reference_counts = _ngrams(reference, order)
        if not candidate_counts or not reference_counts:
            scores.append(0.0)
            continue
        overlap = sum((candidate_counts & reference_counts).values())
        precision = overlap / sum(candidate_counts.values())
        recall = overlap / sum(reference_counts.values())
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(scores) / len(scores)


class CharacterNGramScorer:
    def __init__(self, text: str, order: int = 4, smoothing: float = 0.1):
        self.order = order
        self.smoothing = smoothing
        self.vocabulary = tuple(sorted(set(text)))
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        for index, char in enumerate(text):
            context = text[max(0, index - order + 1) : index]
            self.counts[context][char] += 1

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        log_probability = 0.0
        vocabulary_size = len(self.vocabulary)
        for index, char in enumerate(text):
            context = text[max(0, index - self.order + 1) : index]
            counts = self.counts.get(context, Counter())
            probability = (counts.get(char, 0) + self.smoothing) / (
                sum(counts.values()) + self.smoothing * vocabulary_size
            )
            log_probability += math.log(probability)
        return math.exp(log_probability / len(text))


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    reference_similarity: float
    fluency: float
    eos_length: float
    formatting: float
    anti_repetition: float
    response_length: int
    has_eos: bool


def _formatting_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    checks = (
        stripped[0].isupper() or stripped[0] in "'\"(",
        stripped[-1] in ".?!,:;-'\"",
        ":" not in stripped,
        stripped.count("\n") <= 4,
    )
    return sum(checks) / len(checks)


def _anti_repetition_score(text: str) -> float:
    if len(text) < 3:
        return 0.0
    trigrams = [text[index : index + 3] for index in range(len(text) - 2)]
    distinct_ratio = len(set(trigrams)) / len(trigrams)
    longest_run = 1
    current_run = 1
    for previous, current in zip(text, text[1:]):
        current_run = current_run + 1 if current == previous else 1
        longest_run = max(longest_run, current_run)
    run_score = max(0.0, 1.0 - (longest_run - 2) / 6)
    return 0.75 * distinct_ratio + 0.25 * run_score


def shakespeare_reward(
    response: str,
    reference: str,
    *,
    fluency_scorer: CharacterNGramScorer,
    has_eos: bool,
    generation_length: int,
) -> RewardBreakdown:
    reference_similarity = character_ngram_f1(response, reference)
    fluency = fluency_scorer.score(response)
    response_length = len(response)
    minimum_length = min(16, generation_length)
    upper_length = max(minimum_length, int(generation_length * 0.8))
    if not has_eos:
        eos_length = 0.0
    elif response_length < minimum_length:
        eos_length = 0.5 + 0.5 * response_length / max(1, minimum_length)
    elif response_length <= upper_length:
        eos_length = 1.0
    else:
        eos_length = 1.0 - 0.5 * (response_length - upper_length) / max(
            1, generation_length - upper_length
        )
    formatting = _formatting_score(response)
    anti_repetition = _anti_repetition_score(response)
    total = (
        0.35 * reference_similarity
        + 0.30 * fluency
        + 0.15 * eos_length
        + 0.10 * formatting
        + 0.10 * anti_repetition
    )
    return RewardBreakdown(
        total,
        reference_similarity,
        fluency,
        eos_length,
        formatting,
        anti_repetition,
        response_length,
        has_eos,
    )
