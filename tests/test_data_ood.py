from __future__ import annotations

from datasets import Dataset, DatasetDict

from dapd.data import build_external_eval_dataset


def test_build_external_eval_dataset_prefers_test_split(monkeypatch) -> None:
    train = Dataset.from_dict({"domain": ["x"], "prompt": ["p-train"], "target": ["a-train"]})
    test = Dataset.from_dict(
        {
            "domain": ["x", "x"],
            "prompt": ["p-test-1", "p-test-2"],
            "target": ["a-test-1", "a-test-2"],
        }
    )

    monkeypatch.setattr(
        "dapd.data._load_domain_dataset",
        lambda *args, **kwargs: DatasetDict({"train": train, "test": test}),
    )

    out = build_external_eval_dataset(
        dataset_name="bioasq",
        cache_dir=None,
        seed=42,
        max_eval_samples=1,
    )

    assert len(out) == 1
    assert set(out.column_names) == {"domain", "prompt", "target"}
