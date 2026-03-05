from __future__ import annotations

from pathlib import Path

from dapd.data import _tokenize_cache_file


class DummyTokenizer:
    def __init__(self, name_or_path: str, vocab_size: int) -> None:
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size


def test_tokenize_cache_file_is_deterministic(tmp_path: Path) -> None:
    tok = DummyTokenizer(name_or_path="test/model", vocab_size=32000)

    p1 = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        max_length=256,
        split_name="train",
    )
    p2 = _tokenize_cache_file(
        tokenized_cache_dir=str(tmp_path),
        tokenizer=tok,
        max_length=256,
        split_name="train",
    )

    assert p1 == p2
    assert p1.name.startswith("train_")
    assert p1.suffix == ".arrow"
