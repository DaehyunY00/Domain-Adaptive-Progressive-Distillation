from __future__ import annotations

from pathlib import Path

from datasets import Dataset
import pytest

from dapd.data import _tokenize_cache_file, tokenize_for_causal_lm


class DummyTokenizer:
    def __init__(
        self,
        name_or_path: str = "test/model",
        vocab_size: int = 32000,
        revision: str = "main",
    ) -> None:
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.init_kwargs = {"revision": revision}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        del truncation, add_special_tokens
        # Deterministic toy tokenization for testing cache behavior.
        token_count = max(2, min(max_length, len(text.split()) + 1))
        input_ids = list(range(1, token_count + 1))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


def test_tokenize_cache_file_is_deterministic(tmp_path: Path) -> None:
    tok = DummyTokenizer(name_or_path="test/model", vocab_size=32000, revision="rev-a")
    ds = Dataset.from_dict({"prompt": ["p"], "target": ["t"]})

    p1 = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        dataset=ds,
        dataset_names=["pubmed_qa", "sciq"],
        seed=42,
        max_length=256,
        split_name="train",
        preprocessing_version="v1",
    )
    p2 = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        dataset=ds,
        dataset_names=["pubmed_qa", "sciq"],
        seed=42,
        max_length=256,
        split_name="train",
        preprocessing_version="v1",
    )

    assert p1 == p2
    assert p1.name.startswith("train_")
    assert p1.suffix == ".arrow"


def test_tokenize_cache_file_changes_with_seed_or_preproc_version(tmp_path: Path) -> None:
    tok = DummyTokenizer(name_or_path="test/model", vocab_size=32000, revision="rev-a")
    ds = Dataset.from_dict({"prompt": ["p"], "target": ["t"]})

    seed_a = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        dataset=ds,
        dataset_names=["pubmed_qa"],
        seed=42,
        max_length=256,
        split_name="train",
        preprocessing_version="v1",
    )
    seed_b = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        dataset=ds,
        dataset_names=["pubmed_qa"],
        seed=43,
        max_length=256,
        split_name="train",
        preprocessing_version="v1",
    )
    ver_b = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        dataset=ds,
        dataset_names=["pubmed_qa"],
        seed=42,
        max_length=256,
        split_name="train",
        preprocessing_version="v2",
    )

    assert seed_a != seed_b
    assert seed_a != ver_b


def test_tokenize_for_causal_lm_uses_existing_cache_without_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = DummyTokenizer()
    ds = Dataset.from_dict(
        {
            "prompt": ["question one", "question two"],
            "target": ["answer one", "answer two"],
        }
    )

    first = tokenize_for_causal_lm(
        dataset=ds,
        tokenizer=tok,
        max_length=32,
        num_proc=1,
        split_name="train",
        dataset_names=["pubmed_qa"],
        seed=42,
        preprocessing_version="v1",
        tokenized_cache_dir=str(tmp_path),
        enable_map_cache=True,
    )

    cache_files = list(Path(tmp_path).glob("train_*.arrow"))
    assert len(cache_files) == 1

    def _raise_map(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dataset.map should not be called on cache hit")

    monkeypatch.setattr(Dataset, "map", _raise_map)

    second = tokenize_for_causal_lm(
        dataset=ds,
        tokenizer=tok,
        max_length=32,
        num_proc=1,
        split_name="train",
        dataset_names=["pubmed_qa"],
        seed=42,
        preprocessing_version="v1",
        tokenized_cache_dir=str(tmp_path),
        enable_map_cache=True,
    )

    assert len(first) == len(second)


def test_tokenize_for_causal_lm_preserves_supervised_target_tokens_with_small_max_length(
    tmp_path: Path,
) -> None:
    tok = DummyTokenizer()
    ds = Dataset.from_dict(
        {
            "prompt": ["long prompt " * 40],
            "target": ["final answer"],
        }
    )

    tokenized = tokenize_for_causal_lm(
        dataset=ds,
        tokenizer=tok,
        max_length=8,
        num_proc=1,
        split_name="train",
        dataset_names=["pubmed_qa"],
        seed=42,
        preprocessing_version="v2",
        tokenized_cache_dir=str(tmp_path),
        enable_map_cache=False,
    )

    labels = tokenized[0]["labels"]
    assert len(labels) <= 8
    assert any(label != -100 for label in labels)
